#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/26 10:26
    @Author  : jack.li
    @Site    : 
    @File    : world_company.py

"""
import re
import pandas as pd

world_500_path = "D:\\xxxx\\world_500.csv"
world_500 = pd.read_csv(world_500_path)

usa_500_path = "D:\\xxxx\\usa_500.csv"
usa_500 = pd.read_csv(usa_500_path)

def format_company(input_str):
    return input_str.strip().replace("(", "（").replace(")", "）").upper()
name_dict = dict()
for idx, row in world_500.iterrows():
    if row["country"] != "中国":
        print(row["name"], row["country"])
        short_list = []
        short_name = row["name"].strip().split("（")[0]
        short_list.append(short_name)

        if short_name.lower() not in short_list:
            short_list.append(short_name.lower())
        if short_name.upper() not in short_list:
            short_list.append(short_name.upper())
        if short_name.title() not in short_list:
            short_list.append(short_name.title())

        span = re.search("（(.+?)\)",row["name"])
        if span:
            english_name = span.group(1)
            short_list.append(english_name)
            if english_name.lower() not in short_list:
                short_list.append(english_name.lower())
            if english_name.upper() not in short_list:
                short_list.append(english_name.upper())
            if english_name.title() not in short_list:
                short_list.append(english_name.title())
        format_name = format_company(row["name"])
        name_dict.setdefault(format_name, {"name": format_name, "country": row["country"], "short_list": []})
        name_dict[format_name]["short_list"] = short_list


for idx, row in usa_500.iterrows():
    # if row["country"] != "中国":
    print(row["name"])
    format_name = format_company(row["name"])
    name_dict.setdefault(format_name, {"name": format_name, "country": "美国", "short_list": []})
    # name_dict[format_name]["short_list"] = short_list

    short_name = row["name"].strip().split("(")[0]
    if short_name not in name_dict[format_name]["short_list"]:
        name_dict[format_name]["short_list"].append(short_name)

    if short_name.lower() not in name_dict[format_name]["short_list"]:
        name_dict[format_name]["short_list"].append(short_name.lower())
    if short_name.upper() not in name_dict[format_name]["short_list"]:
        name_dict[format_name]["short_list"].append(short_name.upper())
    if short_name.title() not in name_dict[format_name]["short_list"]:
        name_dict[format_name]["short_list"].append(short_name.title())

    span = re.search("\((.+?)\)", row["name"])
    if span:

        english_name = span.group(1)
        print(english_name)
        if english_name not in name_dict[format_name]["short_list"]:
            name_dict[format_name]["short_list"].append(english_name)
        if english_name.lower() not in name_dict[format_name]["short_list"]:
            name_dict[format_name]["short_list"].append(english_name.lower())
        if english_name.upper() not in name_dict[format_name]["short_list"]:
            name_dict[format_name]["short_list"].append(english_name.upper())
        if english_name.title() not in name_dict[format_name]["short_list"]:
            name_dict[format_name]["short_list"].append(english_name.title())

wolrd_2000_path = "D:\\xxxx\\world2000.csv"
wolrd_2000 = pd.read_csv(wolrd_2000_path)
print("=====================")
for idx, row in wolrd_2000.iterrows():
    if row["country"] != "中国内地":
        # print(row["name"], row["country"])
        format_name = format_company(row["name"])
        if row["name"] in ["SSE", "德科"]:
            name_dict.setdefault(format_name, {"name": format_name, "country": "美国", "short_list": []})
            if row["name"].strip() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip())
            if row["name"].strip().lower() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip().lower())

            if row["name"].strip().upper() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip().upper())

            if row["name"].strip().title() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip().title())
            continue
        if row["name"] == "LG":
            match_name = "LG电子（LG ELECTRONICS）"
            if row["name"] not in name_dict[match_name]:
                name_dict[match_name]["short_list"].append(row["name"].strip())
        if row["name"] == "ABB":
            continue
        state = 0
        for k, v in name_dict.items():
            if row["name"] in k:
                if row["name"] not in name_dict[k]:
                    name_dict[k]["short_list"].append(row["name"].strip())
                # print(row["name"], k)
                state = 1
                break
        if state == 0:
            name_dict.setdefault(format_name, {"name": format_name, "country": row["country"], "short_list": []})
            if row["name"].strip() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip())
            if row["name"].strip().lower() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip().lower())

            if row["name"].strip().upper() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip().upper())

            if row["name"].strip().title() not in name_dict[format_name]["short_list"]:
                name_dict[format_name]["short_list"].append(row["name"].strip().title())

short_dict = dict()
data_list = []
for k, v in name_dict.items():
    for ss in v["short_list"]:
        short_dict[ss] = k
    # k_lower = k
    v["short_list"] = "、".join(v["short_list"])
    data_list.append(v)
print("++++++++++++++++++++++++")
investor_path = "D:\\xxxx\\investor_label_v4.csv"
investor_df = pd.read_csv(investor_path)
iv = 0
for idx, row in investor_df.iterrows():
    if row["investor"] in short_dict:
        iv += 1
print(iv)

#
# import pandas as pd
data_df = pd.DataFrame(data_list)
data_df = data_df.sort_values(by='name')
#
#
# print(data_df.head())
data_df.to_csv("D:\\xxxx\\world_company.csv", index=False)