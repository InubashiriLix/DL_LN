#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :nargs.py
# @Time        :2024/10/8 下午5:23
# @Author      :InubashiriLix

import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('chars',
                    type=str,
                    # required=True,
                    # the require is an invalid argument for the positional
                    nargs=2,
                    metavar='c',
                    help="starting and ending character, the first char should be before the latter one")
args = parser.parse_args()
try:
    # try to convert to string to char
    v1 = ord(args.chars[0])
    v2 = ord(args.chars[1])
except TypeError as e:
    # if the string contains more than one char
    print("Error: Argument must be characters")
    parser.print_help()
    sys.exit(1)
if v1 > v2:
    print('first letter must precede the second in alphabet')
    parser.print_help()
    sys.exit(1)
else:
    for i in range(v1, v2 + 1):
        print(chr(i), end='')
