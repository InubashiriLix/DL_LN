#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :cifar2_dataset.py
# @Time        :2024/10/4 下午10:33
# @Author      :InubashiriLix

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

data_path = r"E:\0-00 PythonProject\TORCH_LEARNING\CHAPTER7\resources\cifar"

cifar10_train_tensor = datasets.CIFAR10(data_path, train=True, download=False,
                                        transform=transforms.ToTensor())
imgs_train = torch.stack([img_single_t for img_single_t, _ in cifar10_train_tensor], dim=3)
mean_train = imgs_train.view(3, -1).mean(dim=1)
std_train = imgs_train.view(3, -1).std(dim=1)
cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(mean_train, std_train)]))

cifar10_val_tensor = datasets.CIFAR10(data_path, train=False, download=False,
                                      transform=transforms.ToTensor())
imgs_val = torch.stack([img_single_t for img_single_t, _ in cifar10_val_tensor], dim=3)
mean_val = imgs_val.view(3, -1).mean(dim=1)
std_val = imgs_val.view(3, -1).std(dim=1)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize(mean_val, std_val)]))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]


# class Cifar2Train(Dataset):
#     def __init__(self, cifar_list):
#         self.data = cifar_list
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#
# class Cifar2Val(Dataset):
#     def __init__(self, cifar_list):
#         self.data = cifar_list
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#
# cifar2_dataloader = Cifar2Train(cifar2)
# cifar2_val_dataloader = Cifar2Val(cifar2_val)
