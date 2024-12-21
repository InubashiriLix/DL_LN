#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :3.7.py
# @Time        :2024/9/24 下午5:33
# @Author      :InubashiriLix
"""
for the STORAGE DIAGRAM and the METADATA
"""

import torch


def title(content: str):
    print("===== " + content + " =====")


title("storage")
a = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
print(a.storage())

title("change storage")
a.storage()[4] = 10  # the original num should be 2
print(a.storage())

title("change storage in the file")
# save and load
# b = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
# torch.save(b, "tensor_to_change_at_index_3.pt")
# print(torch.load("tensor_to_change_at_index_3.pt"))
# # change the tensor
# c = torch.load("tensor_to_change_at_index_3.pt")
# c.storage()[3] = 10
# torch.save(c, "tensor_to_change_at_index_3.pt")
# print(torch.load("tensor_to_change_at_index_3.pt"))

title("Operation at site")
a = torch.ones(2, 3)
a.zero_()
print(a)

title("MATADATA")
print("lets say we have a matrix in 3x3")
tensor1 = torch.tensor([[5, 7, 4], [1, 3, 2], [7, 3, 8]])
print("shape:")
print(tensor1)
print("the element at (1, 1)")
print(tensor1.storage()[0])
print("the element at (2, 1)")
print(tensor1.storage()[0 + 3])
print("the element at (2, 2)")
print(tensor1.storage()[0 + 3 + 1])

title("MATADATA_anotherDiagram")
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
print(second_point.storage_offset())
print(second_point.size())
print(second_point.shape)
print(points.stride())

seconds_point = points[1]
print(second_point.size())
print(second_point.storage_offset())
print(second_point.stride())