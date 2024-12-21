#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :CycleGAN.py
# @Time        :2024/9/22 上午11:49
# @Author      :InubashiriLix

import torch
from ResNetGenerator import ResNetGenerator
from PIL import Image
from torchvision import transforms


netG = ResNetGenerator()
model_path = "ResGFile/horse2zebra_0.4.0.pth"
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

img = Image.open("horse.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

netG.eval()
# print(netG)

batch_out = netG(batch_t)
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_image = transforms.ToPILImage()(out_t)
out_image.save('zebra.jpg')
out_image.show()
