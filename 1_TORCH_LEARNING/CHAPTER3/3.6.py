#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :3,6.py
# @Time        :2024/9/24 下午4:55
# @Author      :InubashiriLix
"""
this is for
THE API OF TENSOR
"""

import torch


def title(content: str):
    print("===== " + content + " =====")


title("transpose API 1")
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
print(a.shape, a_t.shape)

title("transpose API 2")
a = torch.ones(3, 2)
a_t = a.transpose(0, 1)
print(a.shape, a_t.shape)

title("create operation:")
print("create a tensor")
c = torch.ones(1, 1)
d = torch.zeros(1, 1)

title("transpose, index, slice, connect, convert")
print("transpose:")
e = torch.ones(3, 2)
e_t = d.transpose(0, 1)
print((e * e_t).shape)

title("Math Operations")
print("operation according to the points")
f = torch.randn([1, 3])
print(f)
print(torch.abs(f))
print(torch.cos(f))

print("reduction operation")
g = torch.randn([2, 5, 5])
print(g.mean(-3))
print(g.std(-3))
print(g.norm())

print("comparision operation")
print(g.equal(f))
print(g.max())

print("spectrum operation")
print("fuck! I cannot understand")


title("random sample:")
h = torch.randn([3, 3])
print(h)
print(torch.normal(h))

title("Serialized and NonSerialized")
i = torch.randn([4, 4])
torch.save(i, "tensor.pt")
l = torch.load("tensor.pt")
print(l)

title("multithread")
# torch.set_num_threads()

