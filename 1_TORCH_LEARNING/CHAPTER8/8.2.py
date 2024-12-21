# pyright: reportMissingImports=false
# pyright: typeCheckingMode=off
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :8.2.py
# @Time        :2024/10/4 下午11:36
# @Author      :InubashiriLix

# wtf? battle?

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt

from CHAPTER7.cifar2_dataset import cifar2, cifar2_val
from model_self_define_1 import Net, loss_fn, training_loop, evaluation

# from CHAPTER7.cifar2_dataset import cifar2_dataloader, cifar2_val_dataloader


conv = nn.Conv2d(3, 16, kernel_size=3)
print(conv)
# the conv accept three input channel adn output 16 channels
# the size should be [16, 3, 3, 3] while the bias are 16 (same as the output channel)
print(conv.weight.shape, conv.bias.shape)

img, _ = cifar2[0]

# the conv desire a batch of input in size B×C×H×W
output = conv(img.unsqueeze(0))
# print(img.unsqueeze(0).shape, output.shape)
# torch.Size([1, 3, 32, 32]) INPUT
# torch.Size([1, 16, 30, 30]) OUTPUT
# 30 = 32 - 3 + 1

# show the diff between output and input image
# the detach will return a new tensor without the grad tracing !!!
# but the same position in the ram
plt.imshow(output[0, 0].detach(), cmap="gray")
plt.show()
# the H and W are reduced by 2

# padding ! no more saying
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
# padding = (3 - 1) / 2
# the padding there use the reflecting padding, the padding there should be even!!!
# kernel size has two format: (3,3) or 3
# output = conv(img.unsqueeze(0))
# print(img.unsqueeze(0).shape, output.shape)
# torch.Size([1, 3, 32, 32]) torch.Size([1, 1, 32, 32])
# well, this becomes normal as 32 x 32

# to clear all the factors, we put the bias in the kernel 0
# and the weights to be an Normal value
with torch.no_grad():
    conv.bias.zero_()
    conv.weight.fill_(1.0 / 9.0)
    # 1 / 9 mean that the kernal will take the average value of the 9
    # pixels nearby and within the center pixel
    # and output the new tensor

# see what will happend if we do that
# output = conv(img.unsqueeze(0))
# with torch.no_grad():
#     plt.imshow(output[0, 0], cmap='gray')
#     plt.show()
# fuzzy, huh?

# try somthing new
conv_vertical_edges = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv_vertical_edges.bias.zero_()
    conv_vertical_edges.weight[:] = torch.tensor(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
    )

# output_vertical_edges = conv_vertical_edges(img.unsqueeze(0))
# plt.imshow(output[0, 0].detach(), cmap='gray')
# plt.show()

# # to capture the big features like a huge bird that can't be
# # contain in the small kernel
# # we use the max pool or the avg pool to deal with this
# pool_max = nn.MaxPool2d(2)
# # pool_avg = nn.AvgPool2d()

# output = pool_max(img.unsqueeze(0))
# print(img.unsqueeze(0).shape, output.shape)
# plt.imshow(output[0, 0].detach(), cmap='gray')
# plt.show()

# we complete a basic model there

model_test = nn.Sequential(
    # convolution layer
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 8, kernel_size=3, padding=1),
    nn.Tanh(),
    nn.MaxPool2d(2),
    # flatten layer
    nn.Flatten(),
    # full connection layer
    nn.Linear(8 * 8 * 8, 32),
    nn.Tanh(),
    nn.Linear(32, 2),
)

# check the parameters
for name, param in model_test.named_parameters():
    print(name, param.shape)
# or use the numel to count the total
numel_list = [p.numel() for p in model_test.parameters()]
print(numel_list)

# # validate the net
# model = Net()
# output = model(img.unsqueeze(0))
# print(output)

# test for the convnet
convnet_model = Net()
learning_rate = 1e-2
optimizer_SGD = optim.SGD(convnet_model.parameters(), lr=learning_rate)
# loss_fn = nn.CrossEntropyLoss()
cifar_train_loader = DataLoader(cifar2, batch_size=64, shuffle=True)
cifar_val_loader = DataLoader(cifar2_val, batch_size=64, shuffle=False)
training_loop(
    n_epochs=200,
    model=convnet_model,
    optimizer=optimizer_SGD,
    loss_fn=loss_fn,
    train_loader=cifar_val_loader,
)

evaluation(convnet_model, loss_fn=loss_fn, val_loader=cifar_val_loader)
# the result is gt 0.95!!!
# well done, huh?

# now we save our model
data_path = "data/"
torch.save(convnet_model.state_dict(), "birds_vs_airplane.pt")
