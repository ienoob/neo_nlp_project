#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import json
import hashlib
# from pytorch.event_extraction.event_case1.test_finance_model import extractor



import pandas as pd


data_path = "D:\data\\tianjin_dataset\\tj_event\data\\"

document_list = []
# with open("finance_add.json", "w") as f:
#     f.write()

indx = 176
data_file = data_path + "{}.json".format(indx)
try:
    with open(data_file, "r") as f:
        data = f.read()
except UnicodeDecodeError as e:
    with open(data_file, "r", encoding="utf-8") as f:
        data = f.read()

data_dict = json.loads(data)
content = data_dict["content"]
print(content)

for event in data_dict["event"]:
    print(json.dumps(event, indent=4, ensure_ascii=False))

# with open("finance_add.json", "w") as f:
#     f.write(json.dumps(document_list))

    # break

documents = [
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "约3亿美元"
    ],
    "披露时间": [],
    "投资方": [
        "京东"
    ],
    "融资轮次": [],
    "事件时间": [
        "近日"
    ],
    "领投方": [
        "京东"
    ],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
},
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "10亿元"
    ],
    "披露时间": [],
    "投资方": [
        "CMC资本"
    ],
    "融资轮次": [
        "E"
    ],
    "事件时间": [
        "2020年8月"
    ],
    "领投方": [
        "CMC资本"
    ],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
},
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "1亿美元"
    ],
    "披露时间": [],
    "投资方": [
        "五岳资本",
        "经纬中国"
    ],
    "融资轮次": [
        "D"
    ],
    "事件时间": [
        "2019年10月"
    ],
    "领投方": [
    ],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
},
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "4亿元人民币"
    ],
    "披露时间": [],
    "投资方": [
        "eWTP科技创新基金",
        "洪泰基金"
    ],
    "融资轮次": [
        "C"
    ],
    "事件时间": [
        "2019年3月"
    ],
    "领投方": [],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
},
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "7500万元人民币"
    ],
    "披露时间": [],
    "投资方": [
        "经纬中国"
    ],
    "融资轮次": [
        "B"
    ],
    "事件时间": [
        "2018年4月"
    ],
    "领投方": [
        "经纬中国"
    ],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
},
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "1亿元人民币"
    ],
    "披露时间": [],
    "投资方": [
        "璀璨资本",
        "深创投"
    ],
    "融资轮次": [
        "A"
    ],
    "事件时间": [
        "2017年6月"
    ],
    "领投方": [],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
},
{
    "被投资方": [
        "KK集团"
    ],
    "融资金额": [
        "1500万元人民币"
    ],
    "披露时间": [],
    "投资方": [
        "深创投"
    ],
    "融资轮次": [
        "pre-A"
    ],
    "事件时间": [
        "2016年3月"
    ],
    "领投方": [],
    "财务顾问": [],
    "估值": [],
    "trigger": "融资"
}
]
for document in documents:
    for k, v in document.items():
        if isinstance(v, str):
            assert v in content
        else:
            for vi in v:
                # print(vi)
                assert vi in content
# print(type(content))
file_md5_indx = hashlib.md5(content.encode()).hexdigest()
# print(file_md5_indx)
data = {
    "idx": indx,
    "id": file_md5_indx,
    "content": content,
    "event": documents
}

with open("D:\data\\tianjin_dataset\\tj_event\\data\\{}.json".format(indx), "w", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False))
