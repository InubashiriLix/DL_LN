#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :3.10.py
# @Time        :2024/9/24 下午8:55
# @Author      :InubashiriLix

import torch


points = torch.ones(3, 4)
points_np = points.numpy()
print(points_np)

points = torch.from_numpy(points_np)
print(points)