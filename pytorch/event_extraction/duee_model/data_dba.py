#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
path = "D:\\data\\句子级事件抽取\\duee_train.json\\duee_train.json"

with open(path, "r", encoding="utf-8") as f:
    data = f.read()

for item in data.split("\n"):
    if item.strip()=="":
        continue

    item_json = json.loads(item)
    assert len(item_json["text"]) <= 510
    for event in item_json["event_list"]:
        # print(event)
        print(event["trigger"], event["event_type"], event["trigger_start_index"])
        assert item_json["text"][event["trigger_start_index"]:event["trigger_start_index"]+len(event["trigger"])] == event["trigger"]
    print("============================")


    # break
