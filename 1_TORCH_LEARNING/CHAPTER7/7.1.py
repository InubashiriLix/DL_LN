#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :7.1.py
# @Time        :2024/10/4 下午2:59
# @Author      :InubashiriLix
# 7.1 used to validation the torch behaviour in the image set
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms

# download the dataset
data_path = 'resources/cifar/'
# download = False stands for not download the dataset when the file not found
# train = True stands for downloading train dataset
cifar10 = datasets.CIFAR10(data_path, train=True, download=False)
# train = False stands for downloading validation dataset
cifar10_val = datasets.CIFAR10(data_path, train=False, download=False)

# check the parent of this dataset
print(type(cifar10).__mro__)
# (<class 'torchvision.datasets.cifar.CIFAR10'>,
# <class 'torchvision.datasets.vision.VisionDataset'>,
# <class 'torch.utils.data.dataset.Dataset'>,
# <class 'typing.Generic'>, <class 'object'>)

# the dataset inherit from torch.utils.data.dataset.Dataset
# and it has two common methods: __len__, __getitem__
# for __len__
print(len(cifar10))
# for __getitem__
img, label = cifar10[99]
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
print("\n", img, "\n", label, "\n", class_names[label])

# # use the plt to show the image
# plt.imshow(img)
# plt.show()

# how to transforms the PIL data into torch tensor?
# from torchvision import transforms
print(dir(transforms))
# we can find ToTensor obj now

# attention that the ToTensor is an object and you need to
# make an instance of it if you want to use it
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
print(img_t.shape)

# we can also transform the dataset as we use it from initial
cifar10_tensor = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
img1_t, label1 = cifar10_tensor[99]

# print(type(img1_t))
# torch.Tensor
# print(class_names[label1])
# all works

# plt show
# plt is powerful to show the tensor (numpy?)
# plt.imshow(img1_t.permute(1, 2, 0))
# plt.show()

print(img1_t.shape, img1_t.dtype)
# notice that the shape is C H W not H W C
img1_t = img1_t.permute(1, 2, 0)

# the normalization enable the model to avoid the gradient explode and
# provides smaller possibility for 0 gradient
# so we need the normalization to process the datas
# transforms.Normalize()

# because the dataset is small that we can make all the image
# listed in an additional dimension
imgs = torch.stack([img_single_t for img_single_t, _ in cifar10_tensor], dim=3)
# the stack will create a new dimension to contain all thee images
# while the cat can't
# and the dim=3 stands for the position of the new dimension
# shape of the tensor: ([3, 32, 32, 50000])

# calculate the mean on each channel
means = imgs.view(3, -1).mean(dim=1)
# ([0.4914, 0.4822, 0.4465])

# calculate the std
stds = imgs.view(3, -1).std(dim=1)
# ([0.2470, 0.2435, 0.2616])

# use the transformers, we can normalize all the picture
transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      )
        transforms.Normalize(means, stds)
    ])
)

# test_img, test_img_label = transformed_cifar10[99]
# plt.imshow(test_img.permute(1, 2, 0))
# print(class_names[test_img_label])
# plt.show()