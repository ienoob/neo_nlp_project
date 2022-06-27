#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/26 13:13
    @Author  : jack.li
    @Site    : 
    @File    : product_label.py

"""
import os
import re
import pandas as pd

path = "G:\\out\\"

# 品牌
iv = 0
iv_list = []
for file in os.listdir(path):
    file_path = path + file
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    for dt in re.split("[。\n]", data):
        if "平台" in dt:
            iv += 1
            iv_list.append((dt, ""))
            print(dt)
    print("num {}".format(iv))
    if len(iv_list ) >= 1000:
        break
iv_list_df = pd.DataFrame(iv_list, columns=["sentence", "product_label"])
# iv_list_df.to_csv("D:\\xxxx\\brand_1000_v2.csv", index=False)


