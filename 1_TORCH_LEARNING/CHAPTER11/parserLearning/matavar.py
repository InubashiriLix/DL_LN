#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :matavar.py
# @Time        :2024/10/8 下午4:57
# @Author      :InubashiriLix
"""
the metavar indicate how the variable should look like
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v',
                    type=int,
                    default=1,
                    required=True,
                    metavar="int",
                    help="should enter an int")
args = parser.parse_args()

if __name__ == '__main__':
    if args.v:
        print(args.v)