#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :var_args.py
# @Time        :2024/10/8 下午5:34
# @Author      :InubashiriLix
# python var_arg.py 1 3 4
# -> 1(\n) 3(\n) 4(\n)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("nums",
                    type=int,
                    nargs="*"
                    # the * means you can enter infi numbers as positional arguments
                    )
args = parser.parse_args()

for num in args.nums:
    print(num)