#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.7.py
# @Time        :2024/9/30 下午11:32
# @Author      :InubashiriLix
import os
import imageio
import torch
from PIL import Image

# pic list
image_path = r'E:/0-00 PythonProject/TORCH_LEARNING/CHAPTER4/practice/resource'
filenames = [file for file in os.listdir(image_path) if os.path.splitext(file)[1] == '.jpg']
print(filenames)
batch_size = len(filenames)
batch = torch.zeros(batch_size, 3, 255, 255, dtype=torch.uint8)

for i, file in enumerate(filenames):
    img_arr = imageio.imread_v2(os.path.join(image_path, file)).copy()
    img_arr.resize((255, 255, 3))
    # print(img_arr.dtype)
    # print(img_arr.shape)
    img_t = torch.from_numpy(img_arr).permute(2, 0, 1)[:3]
    batch[i] = img_t

# # normalize 1
# batch = batch.float()
# batch /= 255.0

# normalize 2
batch = batch.float()
# for i in range(batch.shape[1]):
#     mean = torch.mean(batch[:, i])
#     print(mean)
#     std = torch.std(batch[:, i])
#     batch[:, i] = (batch[:, i] - mean) / std

# print the mean value of each channel in each picture...
for i in range(batch.shape[0]):
    for j in range(batch.shape[1]):
        mean = torch.mean(batch[i, j])
        print(j, mean)
        std = torch.std(batch[i, j])
        batch[i, j] = (batch[i, j] - mean) / std
