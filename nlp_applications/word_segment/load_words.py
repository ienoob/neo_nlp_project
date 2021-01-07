#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/7 23:32
    @Author  : jack.li
    @Site    : 
    @File    : load_words.py

"""

path = "D:\data\\nlp\\分词_拼音@4万_搜狗.txt"

def load_dic():

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    data_list = data.split("\n")