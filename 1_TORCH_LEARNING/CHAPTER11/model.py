#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :model.py
# @Time        :2024/10/9 下午2:31
# @Author      :InubashiriLix
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LunaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = torch.relu(self.conv(x))
        out = self.max_pool(torch.relu(self.conv2(out)))
        return out


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.tail_batch_norm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 3)
        self.block4 = LunaBlock(conv_channels * 3, conv_channels * 4)

        # TODO: why it is 1152
        self.header_linear = nn.Linear(1152, 2)
        self.header_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # tail
        out = self.tail_batch_norm(x)

        # body
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # flatten
        out = out.view(out.shape[0], -1)

        # header (fc)
        out = self.header_linear(out)

        return out, self.header_softmax(out)

    def _init_weight(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d
            }:
                nn.init.kaiming_normal_(
                    m.weight.data,
                    a=0,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

