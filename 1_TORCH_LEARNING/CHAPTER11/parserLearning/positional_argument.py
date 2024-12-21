#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :positional_argument.py
# @Time        :2024/10/7 下午11:50
# @Author      :InubashiriLix


# the positional argument do not need hte dash line "--"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("age")
args = parser.parse_args()

if args.name and args.age:
    print(f"Hello {args.name}, you're {args.age} now")


