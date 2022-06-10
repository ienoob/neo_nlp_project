#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
from nlp_applications.data_loader import LoaderBaiduDueeFin, EventDocument, Event, Argument, BaseDataIterator, load_json_line_data
data_path = "D:\Work\git\Doc2EDAG\Data\Data\\train.json"
event_set = set()
# with open(data_path, "r", encoding="utf-8") as f:
#     data = f.read()

# json_datas = json.loads(data)
# for jd in json_datas:
#     for et in jd[1]["recguid_eventname_eventdict_list"]:
#
#         event_set.add(et[1])
#
# print(event_set)
# data_path = "D:\data\CCKS 2020：面向金融领域的篇章级事件主体与要素抽取（二）篇章事件要素抽取\ccks4_2_Data\ccks 4_2 Data\event_element_train_data_label.txt"
#
#
# with open(data_path, "r", encoding="utf-8") as f:
#     data = f.read()
#
# for dt in data.split("\n"):
#     if dt.strip() == "":
#         continue
#     for et in json.loads(dt)["events"]:
#         event_set.add(et["event_type"])
#
# print(event_set)
from pytorch.event_extraction.event_model_v1 import g

train_path = "D:\data\篇章级事件抽取\\duee_fin_train.json\\duee_fin_train.json"
dev_path = "D:\data\篇章级事件抽取\\duee_fin_dev.json\\duee_fin_dev.json"

train_data = load_json_line_data(train_path)
dev_data = load_json_line_data(dev_path)

train_json = []
for i, sub_train_data in enumerate(train_data):
    text = sub_train_data["text"]
    title = sub_train_data["title"]
    doc_id = sub_train_data["id"]

    event_list = []
    for sub_event in sub_train_data.get("event_list", []):
        if sub_event["event_type"] == "中标":
            event_list.append(sub_event)
    if len(event_list):
        print(doc_id, title, len(text))
        train_json.append({"content": text, "title": title, "event": event_list})

with open("bidding.json", "w") as f:
    f.write(json.dumps(train_json))
