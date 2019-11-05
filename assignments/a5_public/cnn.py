#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, embed_char_size = 50,  kernel_size = 5, filters = 2, max_word_length = 50):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels = embed_char_size,
                              out_channels = filters,
                              kernel_size = kernel_size,
                              bias = True)
        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.squeeze()
        return x

### END YOUR CODE

