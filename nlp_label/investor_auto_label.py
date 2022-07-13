#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/30 20:52
    @Author  : jack.li
    @Site    : 
    @File    : investor_auto_label.py

"""
import pandas as pd

# data_path = "D:\\xxxx\\invest_company_v2.csv"
#
# data = pd.read_csv(data_path)
#
# company_alias = dict()
# alias_company = dict()
# for idx, row in data.iterrows():
#     short_list = row["short_list"].split("、")
#     # company_alias["company"] =
#     for short in short_list:
#         alias_company.setdefault(short, [])
#         alias_company[short].append(row["name"])
#
# investor_path = "D:\\xxxx\\investor_label_v4.csv"
# df = pd.read_csv(investor_path, low_memory=False)
# # print(df.head())
# ddd = dict()
# n_target_list = []
# path = "G:\download2\\tyc\\"
# iv = 0
# nan_num = 0
# hit = 0
# bit = 0
# import os
# from bs4 import BeautifulSoup
# for idx, row in df.iterrows():
#     # if row["investor"] != "51Talk":
#     #     continue
#     # print(row["investor"], row["match_name"])
#
#     if not isinstance(row["true_name"], float):
#         iv += 1
#         continue
#     if row["investor"] in ddd:
#         hit += ddd[row["investor"]]
#         continue
#     new_match = []
#     if row["investor"] in alias_company:
#         # hit += 1
#         new_match = alias_company[row["investor"]]
#         # continue
#     state = 0
#     file_path = path + row["investor"] + ".html"
#     if os.path.exists(file_path):
#         with open(file_path, "r", encoding="utf-8") as f:
#             html = f.read()
#         soup = BeautifulSoup(html, "html.parser")
#         zz_list = []
#
#         item_list = soup.find_all("a", {"class": "brand sv-search-company-brand"})
#         for item in item_list:
#             span_label = item.find("span", {"class": "tag-common -primary-bg ml8"})
#             if span_label and span_label.text == "投资机构":
#                 print(row["investor"])
#                 company = item.find("span", {"class": "search-company-name hover"})
#                 if company:
#                     # bit += 1
#                     zz_list.append(company.text)
#                     # break
#
#         if zz_list and new_match and zz_list[0] == new_match[0]:
#             state = 1
#             hit += 1
#     row[row["investor"]] = state
#             # if item.text[:5] == "所属公司：":
#             #     print(item.text[5:])
#
#     print(hit, df.shape[0])
def label_v1():
    investee_path = "D:\\xxxx\\investee_label_v4.csv"
    investee_df = pd.read_csv(investee_path, low_memory=False)
    investee_dict = dict()
    for idx, row in investee_df.iterrows():
        if not isinstance(row["true_name"], float):
            continue
        investee_dict[row["investee"]] = row["true_name"]
    print(len(investee_dict))

    investor_path = "D:\\xxxx\\investor_label_v5.csv"
    investor_new_path = "D:\\xxxx\\investor_label_v6.csv"
    df = pd.read_csv(investor_path, low_memory=False)

    kg_path = "D:\\xxxx\\knowledge graph\\investor_kg.csv"
    kg_df = pd.read_csv(kg_path)

    kg_alias_dict = dict()
    for idx, row in kg_df.iterrows():
        short_list = row["short_name"].split("$")
        for short in short_list:
            short_lower = short.lower()
            kg_alias_dict.setdefault(short_lower, [])
            kg_alias_dict[short_lower].append(row["id"])


    def get_score(input_id_list):
        score_list = []
        for idv in input_id_list:
            item = kg_df[kg_df["id"]==idv]
            # print(item.shape)

            sub_source = set(item.iloc[0]["source"].split("$"))
            score = len(sub_source)
            if "tyc" in sub_source:
                score += 2
            score_list.append((idx, item.iloc[0]["full_name"], score))
        score_list.sort(key=lambda x: x[-1], reverse=True)
        return score_list
            # score_list.append(item[0]["source"])
    iv = 1
    i_hit = 0
    hit = 0
    new_df = []
    for idx, row in df.iterrows():
        if not isinstance(row["true_name"], float):
            new_df.append(row)
            iv += 1
            continue
        if row["investor"] in investee_dict:
            row["true_name"] = investee_dict[row["investor"]]
            new_df.append(row)
            i_hit += 1
            continue
        if row["investor"].lower() in kg_alias_dict:
            match_list = kg_alias_dict[row["investor"].lower()]
            score_list = get_score(match_list)
            match_id, match_name, match_score = score_list[0]
            if match_score > 3:
                hit += 1
                print(row["investor"], match_name)
                row["true_name"] = match_name
        new_df.append(row)
            # break
        print(hit, iv, i_hit, idx)

    print(hit+iv+i_hit, df.shape[0], "last {}".format(df.shape[0]-(hit+iv+i_hit)))
    print(len(new_df), df.shape)
    print("end ==================== end")
    assert len(new_df) == df.shape[0]
    # new_df = pd.DataFrame(new_df)
    # new_df.to_csv(investor_new_path, index=False)


if __name__ == "__main__":
    label_v1()