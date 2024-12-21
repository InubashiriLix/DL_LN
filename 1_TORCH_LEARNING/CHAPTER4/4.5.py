#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName    :4.5.py
# @Time        :2024/9/30 下午9:40
# @Author      :InubashiriLix

import torch
import numpy

# import the book Pride and Prejudice data
book_path = 'resources/jane-austen/1342-0.txt'
with open(book_path, 'r', encoding='utf-8') as f:
    text = f.read()
    lines = text.split('\n')
    line = lines[200]  # an example
    print(line)

# create the onehot tensor for the text
letter_t = torch.zeros(len(line), 128)
print(letter_t.shape)
# torch.Size([70, 128])

# onehot start!
for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else ()
    # this will exclude the invalid letter (that is beyond 128 ASCII)
    letter_t[i][letter_index] = 1


# separate the word
def clean_word(input_str: str):
    punctuation = ' .,:;"!?“”_-—'
    word_list = input_str.lower().replace('\n', ' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

# test fot clean_word function
words_in_line = clean_word(line)
print(line, words_in_line)


# give every word a code (total 7260)
# we can find the word's index conveniently
word_list = sorted(set(clean_word(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}
print(len(word2index_dict), word2index_dict)

# we allocate every word in the sentence a onehot value
word_t = torch.zeros(len(words_in_line), len(word2index_dict))
# print(word_t.shape)  # torch.Size([11, 7261])
# 11 is the word counts in the sentence and 7261 is the word dict
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print("{:2} {:4} {}".format(i, word_index, word))


# text embedding
