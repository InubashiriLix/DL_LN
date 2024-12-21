#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :7..2.py
# @Time        :2024/10/4 下午7:59
# @Author      :InubashiriLix


import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from cifar2_dataset import cifar2, cifar2_val

# from cifar2_dataset import cifar2Train, cifar2Val

# # a little test for softmax
# softmax = nn.Softmax(dim=1)  # calculate along the row (dim=1)
# x = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
# print(softmax(x))

# # all connection model:
# n_out = 2
# model = nn.Sequential(
#     # hidden layer 1
#     nn.Linear(3072,
#               512),
#     # activation layer
#     nn.Tanh(),
#     # hidden layer 1 stop
#
#     # output layer
#     nn.Linear(512, n_out),
#     # use the softmax to calculate the possibility
#     nn.LogSoftmax(dim=1)
# )
#
# # make an instance of NILLoss
# loss = nn.NLLLoss()
#
# # # brief view of a bird
# img, label = cifar2[0]
# # plt.imshow(img.permute(1, 2, 0))
# # plt.show()
#
# print(img.shape)
# img_batch = model(img.view(-1).unsqueeze(0))
# print(img.view(-1).unsqueeze(0).shape)
#
# out = model(img.view(-1).unsqueeze(0))
# print(loss(out, torch.tensor([label])))

# to connect the label to the name label
# _, index = torch.max(out, dim=1)
# print the index of the label in the output tensor
# print(index)

# loss function
# Negative Log Likelihood:
# NIL = -sum(log(out_i[c_i]))
# this function will return the negative log likelihood of the output tensor
# while the possibility is very low, it will return a very large number
# when the possibility is lower than 5, it will go down very slowly
# while the possibility is 1, it is 0

# training loop
# model has been defined
# learning_rate = 1e-2
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# loss_fn = nn.NLLLoss()
# n_epoches = 100
# for epoch in range(n_epoches):
#     for img, label in cifar2:
#         out = model(img.view(-1).unsqueeze(0))
#         loss = loss_fn(out, torch.tensor([label]))
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
# the upper training loop is not good because the batch is too large

# a better solution is to use the mini-batch in the SGD
# and it will shuffle the batch after a epoch
# use the class called Dataloder could deal with this shit

# because the fucking DataLoader only accept the dataset
# we need to create one --> see the cifar2_dataset.py

train_loader = DataLoader(
    cifar2,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    cifar2_val,
    batch_size=64,
    shuffle=False
)

# new training loop
# model = nn.Sequential(
#     nn.Linear(3072, 1024),
#     nn.Tanh(),
#     nn.Linear(1024, 512),
#     nn.Tanh(),
#     nn.Linear(512, 128),
#     nn.Tanh(),
#     nn.Linear(128, 2),
#     nn.LogSoftmax(dim=1)
# )

model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()
n_epochs = 100
for epoch in range(1, n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        # print(outputs) :[[-1.3475, -0.3010],
        #                 [-0.1779, -1.8140]]
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

# evaluate the model
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, -1)
        total += batch_size
        correct += int((predicted == labels).sum())
    print("Accuracy: %f", correct / total)
