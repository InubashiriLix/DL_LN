#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :AlexNet.py
# @Time        :2024/9/22 上午10:45
# @Author      :InubashiriLix

from torchvision import models
import torch
from torchvision import transforms
from PIL import Image

# alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)
# print(resnet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("dog1.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)
# print(out)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    print((labels[index[0]], percentage[index[0]].item()))
    print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
