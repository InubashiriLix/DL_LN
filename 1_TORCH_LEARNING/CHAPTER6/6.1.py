#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :6.1.py
# @Time        :2024/10/4 上午11:11
# @Author      :InubashiriLix

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

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
train_t_c_un = t_c[:train_size].unsqueeze(1)
val_t_c_un = t_c[train_size:].unsqueeze(1)
print(train_t_un)
print(val_t_un)

# the 6.2.1 aim to continue the temperature test in 5.1

# loss function
def loss_fn(t_p, t_c) -> torch.tensor:
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

# new model
# setting the linear model (it should be a part of model)
# the attributes in the Linear is in_feature and out_features
# construct the linear layer first
linear_model = nn.Linear(1, 1)
# print(linear_model(val_t_un))

# call the model directly to run forward
y = linear_model(val_t_un)
# do not call forward directly, it will cause the silent error
# z = linear_model.forward(val_t_un)

# check the bias and weight in the model
print(linear_model.weight)
print(linear_model.bias)

# the nn.Model desire a input tensor with the 0 dimension
# is the batch of the input
# like [1, 2, 3, 4, 5, 6] --> [[1], [2], [3], [4], [5], [6]]
x = torch.ones(10, 1)
print(linear_model(x))

# use multi batch to cal is to fully utilize the
# gpu ability to calculate paralleled
# and some high level models use the whole batch processing
# data to optimize...

# and in order to do that, we need to resize the tensors
t_u = t_u.unsqueeze(1)
t_c = t_c.unsqueeze(1)
print(t_u.shape)

# linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(
    linear_model.parameters(),  # input the params from the model
    lr=1e-2
)

print(list(linear_model.parameters()))
# notice that the params are [0.616] for the weight and [asfsa] for the bias,
# the boardcast sys enable the optimizer to apply to the other dimensions in the model

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


# simple test for version1 training_loop
training_loop(
    n_epoches=500,
    optimizer=optimizer,
    model=linear_model,
    t_u_train=train_t_un,
    t_u_val=val_t_un,
    t_c_train=train_t_c_un,
    t_c_val=val_t_c_un
)


