#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.7.2.py
# @Time        :2024/10/1 上午3:37
# @Author      :InubashiriLix

import torch
import numpy

def clean_word(input_str: str):
    punctuation = "[^a-zA-Z0-9_]+"
    word_list = input_str.lower().replace("\n", " ").split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list


text_path = 'resource/text'
with open(text_path, 'r', encoding='utf-8') as f:
    text = f.read()
    lines = text.split('\n')
    line = lines[13]

word_list = sorted(set(clean_word(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
word_in_line = clean_word(line)
line_t = torch.zeros(len(word_in_line), len(word2index_dict))
for i, word in enumerate(word_in_line):
    word_pos = word2index_dict[word]
    line_t[i][word_pos] = 1
