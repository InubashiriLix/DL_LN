#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :8.2_model_self_define_1.py
# @Time        :2024/10/5 下午7:21
# @Author      :InubashiriLix
# this is an temp version for the shit I'll use in the 8.2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datetime

# use the F to use the activation functions
# because the these functions do not need to be called
# so we dont need to save them as attributes

# test for the ckpt for the bird vs airplane ckpt
data_path = r"E:\0-00 PythonProject\TORCH_LEARNING\CHAPTER8\birds_vs_airplane.pt"

loss_fn = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        # the module there (includeing nn.Module if we have one)
        # must be the first attributes
        # and could not be contained in the list or the dict
        # you can use the nn.ModuleList and the nn.ModuleDict
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels // 2, kernel_size=3, padding=1)
        # self.flatten = nn.Flatten()
        # fuck, the book say no to this
        self.fc1 = nn.Linear(self.n_channels // 2 * 8 * 8, out_features=32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # layer 1
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        # layer 2
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        # layer 3 fc 1
        out = torch.tanh(self.fc1(out.view(-1, self.n_channels // 2 * 8 * 8)))
        # to admit the lack of the flatten, we use the view there!!!
        # output layer
        out = self.fc2(out)

        # return !!!
        return out


class NetDropout(nn.Module):
    def __init__(self, n_chan1 = 16):
        super().__init__()
        self.n_chan1 = n_chan1
        self.conv1 = nn.Conv2d(3, self.n_chan1, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(self.n_chan1, self.n_chan1 // 2, kernel_size=3, padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(self.n_chan1 // 2 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # layer 1
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        # layer 2
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2(out)

        # fc1
        out = F.tanh(self.fc1(out.view(-1, self.n_chan1 // 2 * 8 * 8)))
        # fc2
        out = F.tanh(self.fc2(out))
        return out


class NetBatchNorm(nn.Module):
    def __init__(self, n_chan1=16):
        super().__init__()
        self.n_chan1 = n_chan1
        self.conv1 = nn.Conv2d(3, n_chan1, kernel_size=3, padding=1)
        self.conv1_batchNorm = nn.BatchNorm2d(num_features=self.n_chan1)
        self.conv2 = nn.Conv2d(n_chan1, n_chan1 // 2, kernel_size=3, padding=1)
        self.conv2_batchNorm = nn.BatchNorm2d(num_features=self.n_chan1 // 2)
        self.fc1 = nn.Linear(n_chan1 // 2 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1_batchNorm(self.conv1(x))), 2)
        out = F.max_pool2d(torch.tanh(self.conv2_batchNorm(self.conv2(out))), 2)
        out = F.tanh(self.fc1(out.view(-1, self.n_chan1 // 2 * 8 * 8)))
        out = F.tanh(self.fc2(out))
        return out


class NetDepth(nn.Module):
    def __init__(self, n_chan1=32):
        super().__init__()
        self.n_chan1 = n_chan1
        self.n_chan2 = self.n_chan1 // 2  # 16
        self.n_chan3 = self.n_chan2 // 2  # 8

        self.conv1 = nn.Conv2d(3, self.n_chan1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.n_chan1, self.n_chan2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.n_chan2, self.n_chan3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.n_chan3, self.n_chan3, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(self.n_chan3 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # layer 1
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        # layer 2
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        # layer 3
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
        # layer 4
        out = F.max_pool2d(torch.relu(self.conv4(out)) + out, 2)

        out = torch.flatten(out, 1)

        # fc 1
        out = torch.relu(self.fc1(out))
        # fc 2
        out = self.fc2(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, n_chan1):
        super().__init__()
        self.n_chan1 = n_chan1
        self.conv = nn.Conv2d(self.n_chan1, self.n_chan1, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=self.n_chan1)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


class NetResDeep(nn.Module):
    def __init__(self, n_chan1=32, n_blocks=10):
        super().__init__()
        self.n_chan1 = n_chan1
        self.conv1 = nn.Conv2d(3, self.n_chan1, kernel_size=3, padding=1)
        # use the sequential to contain all the res blocks
        self.resBlocks = nn.Sequential(*(n_blocks * [ResBlock(self.n_chan1)]))
        self.fc1 = nn.Linear(self.n_chan1 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), 2)
        out = F.max_pool2d(self.resBlocks(out), 2)
        out = F.relu(self.fc1(out.view(-1, self.n_chan1 * 8 * 8)))
        out = self.fc2(out)
        return out





def training_loop(
        n_epochs: int,
        model,
        optimizer: optim,
        loss_fn,
        train_loader: DataLoader,
        device: torch.device,
        L2_lambda: float,
        use_gpu: bool = False,
):
    # if the device has been set, use it
    if not device:

        if use_gpu:
            device = torch.device('cuda')
            print("training with CUDA !")
        else:
            device = torch.device('cpu')
            print("training with CPU !")

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            if L2_lambda:
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + L2_lambda * l2_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))


def evaluation(
        model,
        loss_fn,
        val_loader: DataLoader,
        device: torch.device,
        use_gpu: bool = False
):
    if not device:
        if use_gpu:
            device = torch.device('cuda')
            print("training with CUDA !")
        else:
            device = torch.device('cpu')
            print("training with CPU !")

    # very important!!!
    # otherwise the model will be in the training mode
    # and the val will always increase   (orz-2)
    with torch.no_grad():

        val_correct = 0
        val_count = 0
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicts = torch.max(outputs, -1)
            val_correct += int((predicts == labels).sum())
            val_count += len(labels)
        print(f'Validation Accuracy: {val_correct} / {val_count}')


def check_device() -> bool:
    if torch.cuda.is_available():
        print("GPU available")
        # print the detail of the gpu
        print(torch.cuda.get_device_properties(0))
        return True
    else:
        print("GPU not available")
        print(torch.cuda.get_device_properties(0))
        return False


if __name__ == '__main__':
    # three ways to view the model
    model_test = Net()

    # numel_list = [p.numel() for p in model.parameters()]
    # print(sum(numel_list), numel_list)
    #
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.shape)
    #
    # for param in model.parameters():
    #     print(param.shape)

    # test for the model
    img1 = torch.randn(3, 32, 32)

    output = model_test(img1.unsqueeze(0))
    print(output)
    print("=== test end ===")
