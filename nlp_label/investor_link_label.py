#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/24 23:39
    @Author  : jack.li
    @Site    : 
    @File    : investor_link_label.py

"""
import os

path = "G:\\out\\"

# file_list = os.listdir(path)
file_path = path + "19a61663b3b66c3da61475bd72f97ab6.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()
    print(data)