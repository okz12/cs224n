#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn.functional as F
import torch.nn as nn
import torch

### YOUR CODE HERE for part 1h
class Highway(nn.Module):

    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(in_features = word_embed_size, out_features = word_embed_size, bias = True)
        self.gate = nn.Linear(in_features = word_embed_size, out_features = word_embed_size, bias = True)

    def forward(self, x):
        x_proj = torch.relu(self.proj(x))
        x_gate = torch.sigmoid(self.gate(x))
        output = torch.mul(x_proj, x_gate) + torch.mul((1 - x_gate), x)
        return output


### END YOUR CODE 

