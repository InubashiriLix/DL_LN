#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :append.py
# @Time        :2024/10/8 下午5:12
# @Author      :InubashiriLix
# the input should like
# python append.py -n Inubashiri -n Aya -name Momiji
# default="Inubashiri", no default !!!!!
# no type!!!!!

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name",
                    # default="Inubashiri", no default !!!!!
                    # no type!!!!!
                    required=False,
                    action="append",
                    dest="names",
                    help="a list of names")
args = parser.parse_args()
# the input should like
# python append.py -n Inubashiri -n Aya -name Momiji

names = args.names
[print(name) for name in names]


