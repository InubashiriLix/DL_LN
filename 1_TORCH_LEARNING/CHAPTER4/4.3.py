#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.3.py
# @Time        :2024/9/27 下午5:51
# @Author      :InubashiriLix

import csv
import numpy
import torch
import pandas as pd

wine_path = 'resources/tabular-wine/winequality-white.csv'

# use numpy
wineq_numpy = numpy.loadtxt(wine_path, dtype=numpy.float32, delimiter=";", skiprows=1)

#implement through pandas
# wine_df = pd.read_csv(wine_path, delimiter=';')
# wineq = wine_df.to_numpy()

# check if all the data has been readed
# with open(wine_path, 'r') as file:
#     reader = csv.reader(file, delimiter=";")
#     for i, row in enumerate(reader):
#         if i >= 5:
#             break
#         print(row)

# convert the numpy array to the pytorch tensor
wineq = torch.from_numpy(wineq_numpy)
# print(wineq.shape, wineq.dtype)

# drop the score from the ori data and put it into the target tensor
data = wineq[:, :-1]  # select the row and col that except for the last row / col
# print(data, data.shape)

# targe 1
# view the score as the continuous ints
target = wineq[:, -1].long()  # accept the score and put it into the unique tensor
# print(target)

# use one-hot encoding
target_onehot = torch.zeros(target.shape[0], 10)  # target.shape[0] stands for how many instances, and 10 stands for how many kinds of scores
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

# process the data
data_mean = torch.mean(data, dim=0)
# print(data_mean)

data_var = torch.var(data, dim=0)
# print(data_var)

data_normalized = (data - data_mean) / torch.sqrt(data_var)
# print(data_normalized)

bad_indexes = target <= 3
# print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())
# print(bad_indexes)  # bad_indexes is a list
mid_indexes = (target > 3) & (target < 7)
good_indexes = target >= 7

bad_data = data[bad_indexes]
mid_data = data[mid_indexes]
good_data = data[good_indexes]

good_mean = torch.mean(good_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
bad_mean = torch.mean(bad_data, dim=0)

col_list = next(csv.reader(open(wine_path), delimiter=';'))
# print(col_list)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
# it seems that the S element exists more than mid and good in the table.
# we can use it to tell the quality of a wine

# check whether we can do that
total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

actual_indexes = target > 5
print(actual_indexes.shape, actual_indexes.type, actual_indexes.sum())
# it seems that the sulfur of the wine is not perfect.

# compare the sulfur
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches / n_predicted, n_matches / n_actual)


