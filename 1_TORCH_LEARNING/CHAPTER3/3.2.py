#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :3.2.py
# @Time        :2024/9/24 上午9:08
# @Author      :InubashiriLix

import torch
from PIL import Image
from torchvision import transforms


# a = torch.ones(3)
# print(a)
# print(a[1])
# print(a[2])

# points = torch.tensor([1, 2, 3])
# print(points[1])
# print(points)

# points = torch.tensor([[1, 2, 3], [2, 3, 4]])
# print(points)
# print(points[1][1])

# points = torch.zeros(3, 2)
# points[2, 1] = 1
# print(points)
# print(points[2: , 1:])  # torch([[1]])!!!

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# img = Image.open("CHARPTER2/dog1.jpg")
# img_t = preprocess(img)
# batch_t = torch.unsqueeze(img_t, 0)

img_t = torch.randn(3, 5, 5)
weight = torch.tensor([0.2126, 0.1752, 0.0722])
batch_t = torch.randn(2, 3, 5, 5)

img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
print(img_gray_naive.shape)
print(batch_gray_naive.shape)

print("==========")
weight_named = torch.tensor([0.2126, 0.1752, 0.0722], names=['channel'])
print(weight_named)

unsqueezed_weight = weight.unsqueeze(-1).unsqueeze_(-1)
img_weight = (img_t * unsqueezed_weight)
batch_weight = (batch_t * unsqueezed_weight)
img_gray_weight = img_weight.sum(-3)
batch_gray_weight = img_weight.sum(-3)
print(batch_weight.shape)
print(batch_t.shape)
print(unsqueezed_weight.shape)

print("==========")

img_named = img_t.refine_names(..., 'channel', 'row', 'columns')
batch_named = batch_t.refine_names(..., 'channel', 'row', 'columns')
print("img name", img_named.shape, img_named.names)
print("batch named", batch_named.shape, batch_named.names)

print("]=====")
weight_aligned = weight_named.align_as(img_named)
print(weight_aligned.shape, weight_aligned.names)

print("============")
gray_named = (img_named * weight_aligned).sum("channel")
print(gray_named.shape)
print(gray_named.names)


print("-====== check if the dimisions are the same")
weight_aligned = weight_named.align_as(img_named)
print(weight_aligned.shape, weight_aligned.names)

gray_named = (img_named * weight_aligned).sum('channel')
print(gray_named.shape, gray_named.names)

print("=====error example ========")
gray_named = (img_named[..., : 3] * weight_named).sum('channel')

