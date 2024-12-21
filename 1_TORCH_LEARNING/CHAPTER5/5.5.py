#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :5.5.py
# @Time        :2024/10/2 上午1:31
# @Author      :InubashiriLix
"""
this file aim to show how to use
auto grad in the torch
"""

import torch
import torch.optim as optim
from matplotlib import pyplot as plt

# original model, loss, training and grad_fn
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c) -> torch.tensor:
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# the requires_grad attribute in the tensor means
# if torch needs to check the functions that do somthing to the tensor
params = torch.tensor([1.0, 1], requires_grad=True)
# every tensor has a grad attribute, and
# in the normal situation, this attribute should be None
# the normal situation means the params has not been calculated
# the grad...
print(params.grad is None)
# so the params has not been calculated
# the grad is None...

# what we need to do is to use loss function to calculate loss and
# use loss tensor backward
loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

# when the loss backward is calculated, the params.grad has valid value
print(params.grad)
# tensor([4620.8970, 84.6000])

# the torch will calculate the derivative of whole loss function chain
# and accumulate the derivative to the params.
# so the next time you want to use the params grad, you
# should set them to 0 to cancel the last accumulation first.
params.grad.zero_()
print(params.grad)


# done

# 5.4.3 iterate the model
# we has find the optimization model preparation, loss_function and model function
# then we can train the model until certain epoch
def training_loop(n_epochs: int,
                  learning_rate: float,
                  params: torch.tensor,
                  t_u: torch.tensor,
                  t_c: torch.tensor,
                  print_params=False) -> torch.tensor:
    """
    @:argument
    :param n_epochs: the num of epoch of training
    :param learning_rate: learning rate
    :param params:
    :param t_u:
    :param t_c:
    :return:
    """
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            # the grad.zero_() won't work for None...
            params.grad.zero_()
        # predict
        t_p = model(t_u, *params)
        # calculate the loss
        loss = loss_fn(t_p, t_c)
        loss.backward()
        # spread backward
        with torch.no_grad():
            # tell the torch do not trace this step on params tensor grad
            params -= params.grad * learning_rate
        # print log
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


# normalization t_u
t_un = t_u * 0.1

# # training!
# params = training_loop(
#     n_epochs=5000,
#     learning_rate=1e-2,
#     params=torch.tensor([1.0, 0.0], requires_grad=True),  # require_grad = True!!!
#     t_u=t_un,
#     t_c=t_c
# )

# optimizer
print(dir(optim))

# optimizer has two methods: zero_grad() and step.zero_grad()
# zero_grad() set all the attribute to zero
# step update the params
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)

# oh, before you put it into the next turn,
# you need set the grad into 0
optimizer.zero_grad()
# by the way, the position of zero_grad() is some kind of
# random, actually, you can put it anywhere before step()

loss.backward()
# the optimizer will update the value in params.grad
# and -= params just like what we do in the training_loop
# all we need to do is to use step
optimizer.step()
print(params)


def training_loop_using_optim(n_epochs: int,
                              learning_rate: float,
                              params: torch.tensor,
                              optimizer: optim,
                              t_u: torch.tensor,
                              t_c: torch.tensor,
                              print_params=False) -> torch.tensor:
    """
    @:argument
    :param n_epochs: the num of epoch of training
    :param learning_rate: learning rate
    :param params:
    :param t_u:
    :param t_c:
    :return:
    """
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


# it is important that the loss.backward() and optimizer has the same params!
params1 = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer_SGD = optim.SGD([params1], lr=learning_rate)
# new_params = training_loop_using_optim(
#     n_epochs=5000,
#     learning_rate=learning_rate,
#     params=params1,
#     optimizer=optimizer_SGD,
#     t_u=t_un,
#     t_c=t_c)

# # use adam optimizer:
# # one feture of adam optimizer is that
# # the lr is self-adaptive
# optimizer_adam = optim.Adam([params1], lr=learning_rate)
# training_loop_using_optim(
#     n_epochs=5000,
#     learning_rate=learning_rate,
#     params=params1,
#     optimizer=optimizer_adam,
#     t_u=t_u,  # because the learning rate in the adam optimizer is self-adaptive, so no test need for normalization t_u
#     t_c=t_c)
# print(params1)


# 5.5.3 train, val and overfitting
# alright, we conclude that we need two steps to get a model:
# 1. trainging to overfitting
# 2. shrink the params until the overfitting disappear.
n_samples = t_u.shape[0]
n_val = int(n_samples * 0.2)

# shuffle the t_u
# use randperm to random the indexes of tensor
shuffled_indices = torch.randperm(n_samples)

# split the train set and val set...
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

print(train_indices, val_indices)

# get the train and val tensor using the indices (consisted of indexes)
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

# normalization t_u
train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

print(train_t_u, val_t_u)


def training_loop_loss(n_epochs: int,
                       optimizer: optim,
                       params: torch.tensor,
                       train_t_u: torch.tensor,
                       val_t_u: torch.tensor,
                       train_t_c: torch.tensor,
                       val_t_c: torch.tensor):
    for epoch in range(1, n_epochs):
        # spread forward
        train_t_p = model(train_t_u, *params)
        loss = loss_fn(train_t_p, train_t_c)
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            loss_val = loss_fn(val_t_p, val_t_c)
            assert not loss_val.required_grad, "loss_val required_grad == TRUE!!! FIX IT!!!"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"train loss: {loss},  val loss: {loss_val}")

    return params


# training and finding losses
params2 = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate2 = 1e-2
optimizer_SGD2 = optim.SGD([params2], lr=learning_rate2)
params2 = training_loop_loss(
    3000,
    optimizer_SGD2,
    params2,
    train_t_un,
    val_t_un,
    train_t_c,
    val_t_c
)