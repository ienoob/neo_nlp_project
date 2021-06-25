#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

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

sentence_role = []
print(train_value[0])
d = dict()
for tv in train_value[0]:
    sentence_role.append(tv.split("\t"))

for sr in sentence_role:
    word = sr[1]
    dep_ind = int(sr[6])
    if dep_ind == 0:
        key = ("root", word)
    else:
        key = (sentence_role[dep_ind-1][1], word)
    d.setdefault(key, 0)
    d[key] += 1

print(d)
# print(sentence_role)
