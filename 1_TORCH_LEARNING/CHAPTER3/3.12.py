#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :3.12.py
# @Time        :2024/9/24 下午9:04
# @Author      :InubashiriLix

import torch
import h5py


points = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
torch.save(points, "ourpoints.t")

with open('ourpoints.t', 'wb') as f:
    torch.save(points, f)

points1 = torch.load('ourpoints.t')
print(points1)

# using h5py
f = h5py.File('ourpoints.hfd5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()

f = h5py.File('ourpoints.hfd5', 'r')
dset1 = f['coords']
last_points = dset1[-2:]

