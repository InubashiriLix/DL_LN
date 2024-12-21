#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :choices.py
# @Time        :2024/10/8 下午5:39
# @Author      :InubashiriLix
# choices

# (AudioSep) E:\0-00 PythonProject\TORCH_LEARNING\CHAPTER11\parserLearning>pyth
# on choices.py --food dick
# fuck you!!!!
#
# (AudioSep) E:\0-00 PythonProject\TORCH_LEARNING\CHAPTER11\parserLearning>python choices.py --food c
# usage: choices.py [-h] [--food {chicken,pickle,dick}]
# choices.py: error: argument --food: invalid choice: 'c' (choose from 'chicken', 'pickle', 'dick')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--food',
                    type=str,
                    choices=['chicken', 'pickle', 'dick'],
                    help="all the food",
                    dest="foods")

args = parser.parse_args()
foods = args.foods

if foods == 'dick':
    print("fuck you!!!!")