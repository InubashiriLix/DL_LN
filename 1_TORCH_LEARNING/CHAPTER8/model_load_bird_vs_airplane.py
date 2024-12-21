#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :model_load_bird_vs_airplane.py
# @Time        :2024/10/5 下午10:29
# @Author      :InubashiriLix
import datetime

import torch
from torch.utils.data import DataLoader
from model_self_define_1 import Net, training_loop, evaluation, loss_fn, data_path, check_device
from CHAPTER7.cifar2_dataset import cifar2, cifar2_val

cifar2_train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
cifar2_val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

# # load the data (to the cpu (default))
# loaded_model = Net()
# loaded_model.load_state_dict(torch.load(data_path))
#
# # evaluate the model
# print("on the val")
# evaluation(model=loaded_model, loss_fn=loss_fn, val_loader=cifar2_val_loader)
# print("on the train ")
# evaluation(model=loaded_model, loss_fn=loss_fn, val_loader=cifar2_train_loader)

# check the device and load the device
if check_device():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# train on the GPU
# model = Net().to(device)
# learning_rate = 1e-2
# optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
# loss_fn = loss_fn
# training_loop(
#     n_epochs=200,
#     optimizer=optimizer,
#     model=model,
#     loss_fn=loss_fn,
#     train_loader=cifar2_train_loader,
#     device=device
# )
#
# evaluation(model=model, loss_fn=loss_fn, val_loader=cifar2_val_loader, device=device)
# evaluation(model=model, loss_fn=loss_fn, val_loader=cifar2_train_loader, device=device)
#
# torch.save(model.state_dict(), data_path)

# load on the GPU:
loaded_model = Net()
# go to the GPU
loaded_model.load_state_dict(torch.load(data_path))
loaded_model.to(device)
evaluation(model=loaded_model, loss_fn=loss_fn, val_loader=cifar2_val_loader, device=device)
evaluation(model=loaded_model, loss_fn=loss_fn, val_loader=cifar2_train_loader, device=device)