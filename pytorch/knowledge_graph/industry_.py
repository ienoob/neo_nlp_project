#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/29 13:39
    @Author  : jack.li
    @Site    : 
    @File    : industry_.py

"""
import pandas as pd
from pytorch.knowledge_graph.industry_kg import industry_dict

industry_path = "D:\\xxxx\\t_industry.csv"
ind = pd.read_csv(industry_path)
for idx, row in ind.iterrows():
    print(row["industry_name"])

def industry_function(input_str):
    if input_str[-2:] == "零售":
        return "零售业"
    return ""



def ff(input_x):
    for k, v in industry_dict.items():
        if input_x in v:
            return True
    return False
import json
path = "D:\\xxxx\\fail_match_industry.json"

with open(path, "r") as f:
    d_json = f.read()
iv = 0
d_dict = json.loads(d_json)
d_dict_list = [(k, v) for k, v in d_dict.items()]
d_dict_list.sort(key=lambda x: x[1])

for k, v in d_dict_list:
    if industry_function(k):
        continue
    if ff(k):
        continue
    print(k, v)
    iv += 1

print(iv)