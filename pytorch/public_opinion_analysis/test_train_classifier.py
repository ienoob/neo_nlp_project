#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from nlp_applications.data_loader import load_json_line_data
train_path = "D:\data\篇章级事件抽取\\duee_fin_train.json\\duee_fin_train.json"
dev_path = "D:\data\篇章级事件抽取\\duee_fin_dev.json\\duee_fin_dev.json"

train_data = load_json_line_data(train_path)
dev_data = load_json_line_data(dev_path)

for data in train_data:
    for event in data.get("event_list", []):
        print(event["event_type"])
