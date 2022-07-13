#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/6/6 21:54
    @Author  : jack.li
    @Site    : 
    @File    : confind_investee.py

"""

import pandas as pd
from pytorch.knowledge_graph.entity_kg import *

def func(input_x):
    if input_x in special_type:
        return "special_type"
    if input_x in duoyu:
        return "duoyu"
    if input_x in right_but_name_not_right:
        return "right_but_name_not_right"
    if input_x in not_complete:
        return "not_complete"
    if input_x in right_cannot_link:
        return "right_cannot_link"
    if input_x in special_entity:
        return "special_entity"
    if input_x in location_entity:
        return "location_entity"
    if input_x in person_entity:
        return "person_entity"
    if input_x in colledge_entity:
        return "colledge_entity"
    if input_x in not_valid_name:
        return "not_valid_name"
    if input_x in time_date:
        return "time_date"
    if input_x in currency_entity:
        return "currency_entity"
    if input_x in multi_entity:
        return "multi_entity"
    if is_not_valid(input_x):
        return "not valid filter function"

    return "unk"

path = "D:\\xxxx\\public_opinion_entity_v2.csv"

df = pd.read_csv(path)
iv = 0

for idx, row in df.iterrows():
    res =  func(row["entity"])
    if res != "unk":
        iv += 1
    print(idx, iv)
    # if row["true_name"] == "x":
    #     iv += 1
    #
    #     print(row["e"], iv)
    #     res = func(row["investee"])
    #     if res != "unk":
    #         print(row["investee"], res)
    #     else:
    #         raise Exception
print(iv)