#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
data_path = "D:\data\句法分析\evsam05\依存分析训练数据\THU\\"

train_path = data_path + "train.conll"

with open(train_path, "r", encoding="utf-8") as f:
    train_data = f.read()

train_value = []
cache = []

train_list = train_data.split("\n")
for row in train_list:
    if row.strip():
        cache.append(row)
    else:
        train_value.append(cache)
        cache = []
if cache:
    train_value.append(cache)

d1 = dict()
d2 = dict()
d3 = dict()
d4 = dict()
for train_row in train_value:
    sentence_role = []
# print(train_value[0])

    for tv in train_row:
        sentence_role.append(tv.split("\t"))

    for sr in sentence_role:
        word = sr[1]
        dep_ind = int(sr[6])
        if dep_ind == 0:
            key1 = ("root", word)
            key2 = ("", word)
            key3 = ("root", sr[3])
            key4 = ("", sr[3])
        else:
            key1 = (sentence_role[dep_ind-1][1], word)
            key2 = (sentence_role[dep_ind-1][3], word)
            key3 = (sentence_role[dep_ind-1][1], sr[3])
            key4 = (sentence_role[dep_ind-1][3], sr[3])
        d1.setdefault(key1, 0)
        d1[key1] += 1

        d2.setdefault(key2, 0)
        d2[key2] += 1

        d3.setdefault(key3, 0)
        d3[key3] += 1

        d4.setdefault(key4, 0)
        d4[key4] += 1


test_sentence = [("我", "rr"), ("每天", "r"), ("都", "d"), ("在", "p"), ("写", "v"), ("程序", "n")]

root_info = [d1.get(("root", word), 0)+d2.get(("", word), 0)+d3.get(("root", pos), 0)+d4.get(("", pos), 0) for word, pos in test_sentence]
print(root_info)

# print(d.get(("root", "都"), 0))
print(len(d1))
# print(sentence_role)

d = dict()
for i, (word1, pos1) in enumerate(test_sentence):
    for j, (word2, pos2) in enumerate(test_sentence):
        if i == j:
            continue
        path = d1.get((word1, word2), 0) + d2.get((pos1, word2), 0) + d3.get((word1, pos2), 0) + d4.get((pos1, pos2), 0)
        d[(i, j)] = path
root_i = np.argmax(root_info)
print(root_i)

cache = [root_i]
connect_list = []
while len(cache) != len(test_sentence):
    max_v = -1
    max_i = -1
    for x in cache:
        for y in range(len(test_sentence)):
            if y in cache:
                continue
            if d[(x, y)] > max_v:
                max_v = d[(x, y)]
                max_i = (x, y)
    cache.append(max_i[1])
    connect_list.append(max_i)

print(connect_list)



