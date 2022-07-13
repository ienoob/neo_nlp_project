#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/27 15:53
    @Author  : jack.li
    @Site    : 
    @File    : investor_institution.py

"""
import pandas as pd
path = "D:\\xxxx\\finance_company_alias.txt"

with open(path, "r", encoding="utf-8") as f:
    data = f.read()
company_dict = dict()
data_list = data.split("\n")
for row in data_list:
    row_list = row.split("\t")
    if len(row_list)<3:
        continue
    name = row_list[0].strip()
    company_dict.setdefault(name, [])
    if row_list[1].strip() and row_list[1].strip() not in company_dict[name]:
        company_dict[name].append(row_list[1].strip())
    if name not in company_dict[name]:
        company_dict[name].append(name)

print(len(company_dict))
company_list = []
for k, v in company_dict.items():
    # company_list.append(k, )
    company_list.append({"name": k, "country": "", "short_list": "、".join(v)})


data_df = pd.DataFrame(company_list)
data_df = data_df.sort_values(by='name')
#
#
# print(data_df.head())
data_df.to_csv("D:\\xxxx\\invest_company_v2.csv", index=False)