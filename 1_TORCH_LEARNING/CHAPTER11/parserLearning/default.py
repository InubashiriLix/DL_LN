#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :default.py
# @Time        :2024/10/8 下午4:44
# @Author      :InubashiriLix

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n",
                    type=int,
                    required=True,
                    default=3,
                    help="how many rounds it requires?")
parser.add_argument('-m',
                    default="Inubashiri",
                    type=str,
                    required=True,
                    help="what's your name?")
parser.add_argument("-p",
                    action="store_true",
                    )

args = parser.parse_args()

if __name__ == '__main__':
    if args.p:
        for i in range(0, args.n):
            print(args.n)
        print(f"your name is {args.m}")

