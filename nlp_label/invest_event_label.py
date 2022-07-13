#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/24 14:38
    @Author  : jack.li
    @Site    : 
    @File    : invest_event_label.py

"""
import os
import pandas as pd
from nlp_label.invest_event_res import event_label
# path = "D:\\xxxx\\invest_event.csv"
path = "D:\\xxxx\\ad_train_data\\"
file_list = os.listdir(path)

for file in file_list:

    if not file.endswith(".txt"):
        continue
    item_id = file.split(".")[0]

    file_path = path + file
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    if "融资" not in data:
        continue
    if item_id in event_label:
        continue
    print(item_id)
    print(data)
    break
