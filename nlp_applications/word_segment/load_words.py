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


load_dic()
