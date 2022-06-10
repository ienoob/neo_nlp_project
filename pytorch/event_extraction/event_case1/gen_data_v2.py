#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
import hashlib
import pandas as pd
from nlp_applications.ner.crf_model import CRFNerModel
data_path = "D:\data\\tianjin_dataset\\tj_event\\trz_events.csv"

data = pd.read_csv(data_path, encoding="gbk")
indx = 433
# 433 第一次更新标注样本
# indx = 106
content = data["content"][indx]


# with open("D:\data\\tianjin_dataset\\tj_event\\data\\{}.json".format(indx), "r") as f:
#     label_data = f.read()
# #
# print(json.dumps(json.loads(label_data), indent=4, ensure_ascii=False))
# model.load_model()
print(content)
# documents = []
dt = 1
# while dt:
#     d = dict()
#     for k in role2id.keys():
#         v = input("input {}:".format(k))
#         if v.strip():
#             d[k] = v
#     if d:
#         documents.append(d)
#     dt = input("is_continue:")


# with open(".json")
documents = [
        {
            "被投资方": ["帝奥微电子"],
            "融资金额": [],
            "披露时间": ["近日"],
            "投资方": [
                "沃衍资本"
            ],
            "融资轮次": [
                "C"
            ],
            "事件时间": [],
            "领投方": [
                "沃衍资本"
            ],
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

# print([content])
print(len(content))
# # #
# with open("D:\data\\tianjin_dataset\\tj_event\\data\\{}.json".format(indx), "w", encoding="utf-8") as f:
#     f.write(json.dumps(data, ensure_ascii=False))

