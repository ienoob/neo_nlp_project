#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/17 23:12
    @Author  : jack.li
    @Site    : 
    @File    : entity_describe_extract.py

"""
from nlp_applications.data_loader import load_json_line_data
data_path = "D:\data\\nlp\语料库\wiki_zh_2019\wiki_zh\AA\\wiki_00"

# with open(data_path, "r", encoding="utf-8") as f:
#     data = f.read()

data_list = load_json_line_data(data_path)
i = 0
for data in data_list:
    print(data)
    i += 1

    if i == 3:
        break
