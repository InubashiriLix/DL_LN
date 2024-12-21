#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.1.py
# @Time        :2024/9/24 下午9:32
# @Author      :InubashiriLix

# import imageio
# import torch
#
# img_arr = imageio.imread_v2('dog1.jpg')
# print(img_arr.shape)
#
# img = torch.from_numpy(img_arr)
# out = img.permute()
#
# import os
# import torch
# import imageio.v2
#
# batch_size = 3
# batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
#
# data_dir = r'/CHAPTER4/resources'
# filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.jpg']
# for i, filename in enumerate(filenames):
#     img_arr = imageio.v2.imread(os.path.join(data_dir, filename))
#     print(os.path.join(data_dir, filename))
#     img_t = torch.from_numpy(img_arr)
#     img_t = img_t.permute(2, 0, 1)
#     img_t = img_t[: 3]
#     batch[i] = img_t
#
# # normalize the data1
#
# # batch = batch.float()
# # batch /= 255.0
#
#
# # normalization 2
#
# n_channel = batch.shape[1]
# for c in range(n_channel):
#     mean = torch.mean(batch[:, 2])
#     std = torch.std(batch[:, 2])
#     batch[:, c] = (batch[:, 2] - mean) / std
#


import imageio
import torch
import os

# 1.
# image_path = 'dog1.jpg'
# img_arr = imageio.imread(image_path)
# img = torch.from_numpy(img_arr)

# 2.

# image_path = r"E:\0-00 PythonProject\TORCH_LEARNING\CHAPTER4\resources"
# filenames = [file for file in os.listdir(image_path) if os.path.splitext(file)[1] == '.jpg']
# batch_size = len(filenames)
# batch = torch.zeros(batch_size, 3, 255, 255, dtype=torch.uint8)
#
# for i, file in enumerate(filenames):
#     img_arr = imageio.imread_v2(file)
#     img_t = torch.from_numpy(img_arr)
#     img_t = img_t.permute(2, 0, 1)
#     img_t = img_t[: 3]
#     batch[i] = img_t
#
#
# # # normalize 1
# # batch = batch.float()
# # batch /= 255.0
#
# # normalize 2
# batch = batch.float()
# n_channels = batch.shape[1]
# for c in range(n_channels):
#     mean = torch.mean(batch[:, c])
#     std = torch.std(batch[:, c])
#     batch[:, c] = (batch[:, c] - mean) / std

# if any error: see practice 4.7.1
