#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :3.8.py
# @Time        :2024/9/24 下午8:12
# @Author      :InubashiriLix

import torch

points = torch.tensor([[4.0, 1.0], [5, 3], [2, 1]])
print(points)

points_t = points.t()
print(points_t)

print("compare if the transposed share the same storage")
# so the transposed matrix will share the same storage
print(id(points_t.storage()) == id(points.storage()))

print("compare the strides")
print(points.stride())
print(points_t.stride())
# so the new transposed (ues t() only) will not create new storage and
# it will use different stride by jumping


print("transpose with high dimension")
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
print(some_t.shape)
# output 3, 4, 5
print(transpose_t.shape)
# output 5, 4, 3

print(some_t.stride())
print(transpose_t.stride())

print(some_t.is_contiguous())
print(transpose_t.is_contiguous())  # the transpose is not continous

points = torch.tensor([[4, 1], [5, 3], [2, 1]])
points_t = points.t()
print(points)
print(points_t)
print(points_t.storage())
print(points.storage())
# the are the same
print(points_t.stride())

# make the transpose continuous
points_t_cont = points_t.contiguous()
print(points_t_cont)

print("using gpu")
points_gpu = torch.tensor([[4, 1], [5, 3], [2, 1]], device='cuda')
print(points_gpu)
points_gpu = points_gpu + 4

# points_cpu = points_gpu.to(device='cpu')
points_cpu = points_gpu.cpu()
points_gpu_1 = points_cpu.gpu()

