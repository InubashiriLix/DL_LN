#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.2.py
# @Time        :2024/9/27 下午5:45
# @Author      :InubashiriLix

import imageio
import torch

dir_path = 'resources/volumetric-dicom/2-LUNG 3.0  B70f-04083'
vol_arr = imageio.volread(dir_path, 'DICOM')
vol_t = torch.from_numpy(vol_arr)
vol_t = torch.unsqueeze(vol_t, 0)
print(vol_t.shape)
