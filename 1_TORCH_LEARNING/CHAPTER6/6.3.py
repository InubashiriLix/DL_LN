#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :6.3.py
# @Time        :2024/10/4 下午12:52
# @Author      :InubashiriLix

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# original model, loss, training and grad_fn
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

whole_un = t_c * 0.1

# define the train_set and val_set
train_size = int(len(t_c) * 0.8)
shuffled_indexes = torch.randperm(t_c.shape[0])
train_t_un = whole_un[:train_size].unsqueeze(1)
val_t_un = whole_un[train_size:].unsqueeze(1)
train_t_c = t_c[:train_size].unsqueeze(1)
val_t_c = t_c[train_size:].unsqueeze(1)

linear_model = nn.Linear(1, 1)

# linear_model = nn.Linear(1, 1)
optimizer_SGD = optim.SGD(
    linear_model.parameters(),  # input the params from the model
    lr=1e-2
)


def training_loop(
        n_epoches: int,
        optimizer: optim,
        model: nn.Module,
        t_u_train: torch.tensor,
        t_u_val: torch.tensor,
        t_c_train,
        t_c_val
):
    for epoch in range(1, n_epoches):
        t_p = model(t_u_train)
        loss_fn = nn.MSELoss()
        loss = loss_fn(t_p, t_c_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_t_p = model(t_u_val)
            val_loss = loss_fn(val_t_p, t_c_val)
            # assert val_loss.requires_grad, "loss_val required_grad == TRUE!!! FIX IT!!!"
        print(f"train loss: {loss}, val loss: {val_loss}")


# # simple test for version1 training_loop
# training_loop(
#     n_epoches=500,
#     optimizer=optimizer_SGD,
#     model=linear_model,
#     t_u_train=train_t_un,
#     t_u_val=val_t_un,
#     t_c_train=train_t_c_un,
#     t_c_val=val_t_c_un
# )
# not now~~

# # continue from the 6.1, 6,2
# # torch provides a container called nn.Sequential
# seq_model = nn.Sequential(
#     nn.Linear(1, 13),
#     nn.Tanh(),
#     nn.Linear(13, 1)
# )
#
# # use the seq can print the model detail including the layers
# print(seq_model)
#
# # check the paramters in the models
# print([param.shape for param in seq_model.parameters()])
# for name, param in seq_model.named_parameters():
#     print(name)
#     print(param)


# use orderedDict to give layer a name
# use OrderedDict
seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))
print(seq_model)
# Sequential(
#   (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
#   (hidden_activation): Tanh()
#   (output_linear): Linear(in_features=8, out_features=1, bias=True)
# )

for name, param in seq_model.named_parameters():
    print(name, param.shape)

# # use name in the orderedDict to get the certain param
# output_bias = seq_model.output_linear.bias
# output_weight = seq_model.output_linear.weight
# print(output_weight)
# print(output_bias)

optimizer_SGD = optim.SGD(
    seq_model.parameters(),  # input the params from the model
    lr=1e-3
)

training_loop(
    n_epoches=5000,
    optimizer=optimizer_SGD,
    model=seq_model,
    t_u_train=train_t_un,
    t_u_val=val_t_un,
    t_c_train=train_t_c,
    t_c_val=val_t_c
)
# after the training, the seq_model has save all the data in the params

print("output", seq_model(val_t_un))
print("answer", val_t_c)
print("hidden", seq_model.hidden_linear.weight.grad)

# from matplotlib import pyplot as plt
#
# t_range = torch.arange(20., 90.).unsqueeze(1)
#
# fig = plt.figure(dpi=600)
# plt.xlabel("F")
# plt.ylabel("C")
# plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
# plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
# plt.show()
