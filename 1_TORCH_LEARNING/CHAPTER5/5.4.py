#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :5.4.py
# @Time        :2024/10/1 下午5:53
# @Author      :InubashiriLix

import torch
import numpy

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


w = torch.ones(())
b = torch.zeros(())
t_p = model(t_u, w, b)
loss = loss_fn(t_p, t_c)
print(loss)

# give a little change to the w to see the change
delta = 0.01
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w, b), t_c))
print(loss_rate_of_change_w)  # 481.1377


# the steps of MATH using the gradient descent...
# if we want to find the changing rate of w in the lossing function line:
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


# then, we get the changing rate of loss on w and b
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


# 5.4.3 iterate the model
# we has find the optimization model preparation, loss_function and model function
# then we can train the model until certain epoch
def training_loop(n_epochs, learning_rate, params: torch.tensor, t_u, t_c, print_params=False) -> torch.tensor:
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
        w, b = params

        t_p = model(t_u, w, b)  # spread forward
        loss = loss_fn(t_p, t_c)  # calculate the loss
        grad = grad_fn(t_u, t_c, t_p, w, b)  # spread backward
        params = params - learning_rate * grad  # change the params like w, b
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        if print_params:
            print("w: ", w, "   b: ", b)
            print(params)

    return params


# training step
# the learning_rate too big -> waving to death
# the learning_rate big -> unstable
# the learning_rate small -> stable
# too small -> cost too much time
training_loop(
    n_epochs=100,
    learning_rate=1e-4,
    params=torch.tensor([1.0, 0.0]),
    t_u=t_u,
    t_c=t_c,
    print_params=True
)
# if the learning rate is 1e-3
# the loss will be stable at epoch 6 with result 29
# and seems no more optimization then
# we can apply self-adapt learning_rate to solve it

# normalization the data so that
# the learning rate can be apply to all the
# weights / bias
# normalization means put all the data to the -1 ~ 1
t_un = 0.1 * t_u  # in this instance, you can do that to put all the data

# the normalization enable the params will not explode in the training
# which means we can use a relatively big learning rate
params = training_loop(
    n_epochs=200,
    learning_rate=1e-2,  # 1e-4 --> 1e-2
    params=torch.tensor([1.0, 0.0]),
    t_u=t_un,
    t_c=t_c,
    print_params=False
)

# visualize the data
from matplotlib import pyplot as plt

t_p = model(t_un, *params)

fig = plt.figure(dpi=600)
plt.xlabel("temperature F")
plt.ylabel("Temperature C")
# plt accept the numpy array so that we need to
# transform the tensor to the numpy tensor
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()
