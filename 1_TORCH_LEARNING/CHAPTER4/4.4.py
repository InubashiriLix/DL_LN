#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.4.py
# @Time        :2024/9/30 下午5:15
# @Author      :InubashiriLix

import numpy as np
import pandas as pd
import torch
import csv

# import the data first
data_path = 'resources/bike-sharing-dataset/hour-fixed.csv'
bikes_numpy = np.loadtxt(data_path, delimiter=',', dtype=np.float32, skiprows=1,
                         converters={1: lambda x: float(x[8:10])})
# converter converts the data in the list index 1 into the lambda function and slice into float
# like 2020-01-01 --> 1.0
bikes = torch.from_numpy(bikes_numpy)
print(bikes)
# check the headers:
print(next(csv.reader(open(data_path), delimiter=',')))

# check the same tensor with 24h interval
# the data should be in the shape of [hour, list] shape
print(bikes.shape, bikes.stride())
# torch.Size([17520, 17]) (17, 1)

# convert the data into [day, hours, 17 lists] shape
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape, daily_bikes.stride())
# (torch.Size([730, 24, 17]), (408, 17, 1))
# -> N batch, L days, C channels(attibutes/fetures), to convention...

# the model like LSTM, RNN needs the NCL like data
# to get N, C, L, we need transpose
daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())
# torch.Size([730, 17, 24]) (408, 1, 17)

# connect the weather condition to the onehot
# and put the new weather back.
first_day = bikes[:24].long()  # choose the first day in the original tensor as an sample
print(first_day.shape)  # torch.Size([24, 17])
weather_onehot = torch.zeros(first_day.shape[0], 4)  # shape like (24, 4)
print(first_day[:, 9])  # the object weather list

weather_onehot.scatter_(
    1,  # the channel dim in the daily_bikes, shape the (dim1, dim2) like (17, 24)
    first_day[:, 9].unsqueeze(1).long() - 1,
    1.0
)
# check the weather data
print(weather_onehot)

# connect the onehot metrix to the original matrix
print(torch.cat((bikes[:24], weather_onehot), 1)[:1])

# do the upper steps to the daily_bikes
daily_weather_onehot = torch.zeros(daily_bikes.shape[0],
                                   4,
                                   daily_bikes.shape[2])
print(daily_weather_onehot.shape)
# torch.Size([730, 4, 24])

daily_weather_onehot.scatter_(
    1, daily_bikes[:, 9, :].long().unsqueeze(1) - 1, 1.0)

# connect the onehot to the daily_bike
daily_bike = torch.cat((daily_bikes, daily_weather_onehot), 1)
print(daily_bike.shape)
# torch.Size([730, 17, 24]) -> torch.Size([730, 21, 24])
# cat append the 4 list in the onehot to the end of the daily_bike dim 1

# another way to process the weather data
# it calculate the weather data and put it into [0 - 1]
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0
# do it to the temperature list
temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min))
