#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :7.2_recite.py
# @Time        :2024/10/4 下午11:15
# @Author      :InubashiriLix

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets
import torchvision.transforms as transforms


# import the original data
data_path = 'resources/cifar'
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']

cifar10 = torchvision.datasets.CIFAR10(data_path, train=True, download=False,
                                       transform=transforms.ToTensor())
cifar10_train_imgs = torch.stack([img_t for img_t, label in cifar10], 3)
cifar10_train_maen = cifar10_train_imgs.view(3, -1).mean(dim=1)
cifar10_train_std = cifar10_train_imgs.view(3, -1).std(dim=1)
cifar10_normalized = torchvision.datasets.CIFAR10(data_path, train=True, download=False,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(cifar10_train_maen, cifar10_train_std)
                                                  ]))
cifar2_train = [(img_t, label_map[label]) for img_t, label in cifar10_normalized if label in label_map.keys()]
cifar2DataLoader = DataLoader(
    cifar2_train,
    batch_size=64,
    shuffle=True
)

cifar10_val = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                                           transform=transforms.ToTensor())
cifar10_val_imgs = torch.stack([img_t for img_t, label in cifar10_val], 3)
cifar10_val_mean = cifar10_val_imgs.view(3, -1).mean(dim=1)
cifar10_val_std = cifar10_val_imgs.view(3, -1).mean(dim=1)
cifar10_val_normalized = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(cifar10_val_mean, cifar10_val_std)
                                                      ]))
cifar2_val = [(img_t, label_map[label]) for img_t, label in cifar10_val_normalized if label in label_map.keys()]
cifar2ValDataLoader = DataLoader(
    cifar2_val,
    batch_size=64,
    shuffle=False
)


def training(
        epochs_num: int,
        model,
        loss_fn,
        optimizer,
        train_loader: DataLoader,
):
    for epoch_index in range(1, epochs_num+1):
        for imgs, labels in train_loader:
            batch_size = imgs.shape[0]
            # spread forward
            predicts = model(imgs.view(batch_size, -1))
            loss = loss_fn(predicts, labels)
            # spread backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: %d, Loss: %f" % (epoch_index, float(loss)))


def evaluation(
        model,
        loss_fn,
        val_loader: DataLoader,
):
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, lables in val_loader:
            batch_size = imgs.shape[0]
            output = model(imgs.view(batch_size, -1))
            _, predicts = torch.max(output, -1)
            # forward only
            loss = loss_fn(predicts, lables)

            total += batch_size
            correct += int((predicts == lables).sum())

        print(f"accuracy: {correct / total}")


# train the model
model = nn.Sequential(
    nn.Linear(32 * 32 * 3, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)

loss_fn = nn.NLLLoss()
learning_rate = 1e-3
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

training(
    50,
    model,
    loss_fn,
    optimizer,
    cifar2DataLoader
)

evaluation(model, loss_fn, cifar2ValDataLoader)
