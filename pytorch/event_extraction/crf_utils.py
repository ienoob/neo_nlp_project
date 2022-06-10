#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

def split(input_str):
    split_char = {"。", "\n", "\r", "；"}
    not_add_char = {"\r", "\n"}
    start = 0
    for i, i_char in enumerate(input_str):
        if i_char not in split_char:
            continue
        if i > start:
            if i_char in not_add_char:
                sub_str = input_str[start:i]
            else:
                sub_str = input_str[start:i+1]
            sub_str = sub_str.strip()
            yield sub_str
        start = i+1
    if start<len(input_str):
        sub_str = input_str[start:]
        yield sub_str

class DSU(object):
    def __init__(self, N):
        self.root = [i for i in range(N)]
        self.depth = [1 for i in range(N)]

    def find(self, k):
        if self.root[k] == k:
            return k
        return self.find(self.root[k])

    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        xh = self.depth[x]
        yh = self.depth[y]
        if x == y:
            return
        if xh >= yh:
            self.root[y] = x
            self.depth[x] = max(self.depth[x], self.depth[y] + 1)
        else:
            self.root[x] = y
