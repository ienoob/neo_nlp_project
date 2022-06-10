#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json


with open("finance.json", "r") as f:
    datas = f.read()

with open("finance_add.json", "r") as f:
    datas_add = f.read()


datas = json.loads(datas)
datas_add = json.loads(datas_add)
datas += datas_add
i_company = dict()
for data in datas:
    text = data["text"]
    # pattern_list = ["（简称“([\u4e00-\u9fa5]+?)”）",
    #                 "（以下简称：([\u4e00-\u9fa5]+?)）",
    #                 "（以下简称“([\u4e00-\u9fa5]+?)”）",
    #                 "\(以下简称([\u4e00-\u9fa5]+?)\)",
    #                 "\(简称：([\u4e00-\u9fa5]+?)\)",
    #                 "（简称([\u4e00-\u9fa5]+?)）",
    #                 "（下文简称“([\u4e00-\u9fa5]+?)”）",
    #                 "\(下文简称“([\u4e00-\u9fa5]+?)”\)",
    #                 "（公司简称：([\u4e00-\u9fa5]+?)）",
    #                 "（以下简称：([\u4e00-\u9fa5]+?),\d{6}.SH）",
    #                 "（简称“([\u4e00-\u9fa5]+?)”，股票代码\d{6}）",
    #                 "\(以下简称“([\u4e00-\u9fa5]+?)”\)",
    #                 "\(简称“([\u4e00-\u9fa5]+?)”\)",
    #                 "（以下简称“([\u4e00-\u9fa5]+?)”或“公司”）",
    #                 "（公司简称：([\u4e00-\u9fa5]+?)，证券代码：\d{6}.SZ）"]
    pattern_list = [
        "公司全资子公司([\u4e00-\u9fa5]{5-17})（简称“([\u4e00-\u9fa5]+?)”）"
    ]


    if "简称" in text:
        print(data["id"])
        print(text)
        # print("简称")
        for pattern in pattern_list:
            res = re.finditer(pattern, text)
            for r in res:
                print(r.span(), r.span(1))
    #
    #
    #     print(data["event"])

    # for event in data["event"]:
    #     for arg in event["arguments"]:
    #         if arg["role"] in ["投资方"]:
    #             print(arg["argument"])
                # i_company.setdefault(arg["argument"], 0)
                # i_company[arg["argument"]] += 1

# i_company_list = [(k, v) for k, v in i_company.items()]
# i_company_list.sort(key=lambda x: x[1], reverse=True)
#
# print(i_company_list[:20])
