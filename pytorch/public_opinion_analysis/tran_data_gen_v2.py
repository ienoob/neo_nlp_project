#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
data_path = "train.json"

with open(data_path, "r") as f:
    data = f.read()

data_list = json.loads(data)
iv = 0
print("idx {}".format(iv))
event_type_set = set()
for dt in data_list[iv:500]:

    # print(dt["data_item"])
    # if dt["data"]["id"] != "4a6617e4d1077ba3f6d509aa237741b3":
    #     continue
    print(dt["data"]["id"])
    seq_label = []
    for i, it in enumerate(dt["data_item"]):


        sentence = it["sentence"]
        print(i, sentence)

        label_value = ["O"]*len(sentence)
        for label_item in it["label"]:
            label_value[label_item["start"]] = "{}-S".format(label_item["type"])
            for si in range(label_item["start"]+1, label_item["end"]):
                label_value[si] = "{}-I".format(label_item["type"])
    #     sq_pair = ["{}\t{}".format(itm[0], itm[1]) for itm in zip(sentence, label_value)]
    #     print(sq_pair)
    #     seq_label.append("\n".join(sq_pair))
    #
    # with open("D:\data\舆情分析\self-label\{}.txt".format(dt["data"]["id"]), "w", encoding="utf-8") as f:
    #     f.write("\n======================\n".join(seq_label))

    # print(dt["data"])
    # print(dt["data"]["title"])
    for event in dt["data"].get("event_list", []):
        print(event)
        event_type_set.add(event["event_type"])

    break
print(event_type_set)
