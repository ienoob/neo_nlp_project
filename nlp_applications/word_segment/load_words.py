#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/7 23:32
    @Author  : jack.li
    @Site    : 
    @File    : load_words.py

"""

path = "D:\data\\nlp\\分词\\199801\\199801.txt"


def load_dic():

    with open(path, "r", encoding="gbk") as f:
        data = f.read()

    data_list = data.split("\n")

    words_dict = dict()
    for data in data_list:
        if len(data.strip()) == 0:
            continue
        d = data.split(" ")
        for x in d[1:]:
            if len(x) == 0:
                continue
            x = x.split("/")[0]
            words_dict.setdefault(x, 0)
            words_dict[x] += 1
    return words_dict


# load_dic()



def generator_seg_sentence(input_data_path=None):
    with open(input_data_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    data_list = []
    cache = []
    max_len = 0
    for train_row in train_data.split("\n"):
        train_row = train_row.strip()
        if train_row == "":
            seg_data = [x[1] for x in cache]
            raw_data = "".join(seg_data)
            max_len = max(max_len, len(raw_data))

            data_list.append({"seg_data": seg_data, "raw_data": raw_data})
            cache = []
        else:
            train_row_dep = train_row.split("\t")
            assert len(train_row_dep) == 8
            cache.append(train_row_dep)
    if len(cache) > 1:
        seg_data = [x[1] for x in cache]
        raw_data = "".join(seg_data)
        max_len = max(max_len, len(raw_data))

        data_list.append({"seg_data": seg_data, "raw_data": raw_data})
    return data_list, max_len




