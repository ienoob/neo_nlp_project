#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/6/6 21:54
    @Author  : jack.li
    @Site    : 
    @File    : confind_investee.py

"""

import pandas as pd
from pytorch.knowledge_graph.entity_kg import is_not_valid


path = "D:\\xxxx\\investee_label_v4.csv"

df = pd.read_csv(path)
iv = 0
for idx, row in df.iterrows():
    if row["true_name"] == "x":
        iv += 1

        print(row["investee"], iv)
        res = is_not_valid(row["investee"])
        # print(res)
        if res:
            print(row["investee"], res)
        else:
            raise Exception