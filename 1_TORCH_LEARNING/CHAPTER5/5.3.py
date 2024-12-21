#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :5.3.py
# @Time        :2024/10/1 下午5:48
# @Author      :InubashiriLix

import torch
x = torch.ones(())
y = torch.ones(3, 1)
z = torch.ones(1, 3)
a = torch.ones(2, 1, 1)

print(f"shape: {x.shape}, y: {y.shape}")
print("x * y: ", (x * y).shape)
print("y * z: ", (y * z).shape)
print("y * z * a: ", (y * z * a).shape)
