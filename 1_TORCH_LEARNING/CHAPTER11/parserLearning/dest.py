#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :dest.py
# @Time        :2024/10/7 下午11:56
# @Author      :InubashiriLix

import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-n",

                    dest="now",
                    # the dest will use the variable that has the same name in the
                    # following code

                    action="store_true",
                    # the action there means that this param will work like an
                    # trigger (only when -n and store_true"

                    help="show the current time"
                    )
args = parser.parse_args()

if __name__ == '__main__':
    if args.now:
        now = datetime.datetime.now()
        print(f"now : {now}")
# example
# if you enter the -n, the program will cause the now automatically
# the value in the param is not desired....