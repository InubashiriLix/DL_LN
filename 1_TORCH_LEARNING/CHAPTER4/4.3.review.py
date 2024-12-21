#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.3.review.py
# @Time        :2024/9/29 下午12:08
# @Author      :InubashiriLix

import csv
import torch
import numpy as np
import pandas as pd

wine_path = 'resources/tabular-wine/winequality-white.csv'

# use numpy
wineq_numpy = np.loadtxt(wine_path, delimiter=';', skiprows=1, dtype=np.float32)
# use pandas
# wineq_numpy1 = pd.read_csv(wine_path, delimiter=';')
# wineq = wineq_numpy1.to_numpy()

# check if all the data has been read
# with open(wine_path, 'r') as f:
#     reader = csv.reader(f, delimiter=';')
#     for i, row in enumerate(reader):
#         if i >= 5:
#             break
#         print(row)
# or
print(next(csv.reader(open(wine_path, 'r'), delimiter=';')))

# put the data into the tensor

wineq = torch.from_numpy(wineq_numpy)

data = wineq[:, :-1]
target = wineq[:, -1].long()
# print(target)

target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)
data_normalized = (data - data_mean) / torch.sqrt(data_var)

bad_indexes = target <= 3
mid_indexes = (target > 3) & (target < 7)
good_indexes = (target >= 7)

bad_data = data[bad_indexes]
mid_data = data[mid_indexes]
good_data = data[good_indexes]

good_mean = torch.mean(good_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
bad_mean = torch.mean(bad_data, dim=0)

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print(col_list)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
