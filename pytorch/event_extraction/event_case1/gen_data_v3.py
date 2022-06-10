#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import json
import hashlib
# from pytorch.event_extraction.event_case1.test_finance_model import extractor



import pandas as pd
# data_path = "D:\data\\tianjin_dataset\\tj_event\\trz_events.csv"
#
# data = pd.read_csv(data_path, encoding="gbk")
# idx = 8
# for content in data["content"][idx:]:
#     if isinstance(content, float):
#         continue
#     print("idx {}".format(idx))
#     print(content)
#     print(extractor(content))
#
#     break


data_path = "D:\data\\tianjin_dataset\\tj_event\data\\"

document_list = []
# with open("finance_add.json", "w") as f:
#     f.write()
iv = 0
for file in os.listdir(data_path):
    data_file = data_path + file
    try:
        with open(data_file, "r") as f:
            data = f.read()
    except UnicodeDecodeError as e:
        with open(data_file, "r", encoding="utf-8") as f:
            data = f.read()

    data_dict = json.loads(data)

    # print(data_dict["content"])
    # print(data_dict["event"])

    n_event_list = []
    for event in data_dict["event"]:
        n_event = {"trigger": "", "arguments": []}
        for k, v in event.items():
            if v == "" or v == []:
                continue
            if k == "trigger":
                n_event["trigger"] = v
            else:
                if isinstance(v, str):
                    n_event["arguments"].append({"argument": v, "role": k})
                    # n_event[k] = [v]
                else:
                    for vi in v:
                        n_event["arguments"].append({"argument": vi, "role": k})

                if k == "披露时间":
                    print(file)
                    print(v)
        n_event_list.append(n_event)

    new_event = {
        "idx": int(file.split(".")[0]),
        "id": data_dict["id"],
        "text": data_dict["content"],
        "event": n_event_list
    }
    if isinstance(data_dict["id"], int):
        new_event["id"] = hashlib.md5(data_dict["content"].encode()).hexdigest()
    # print(new_event["idx"])
    print(new_event["id"])
    print(len(data_dict["content"]))

    # if new_event["id"] == "3cfb6bd78c6acbdc611aa37a06bac1b6":
    #     print(file, "+++++++++")

    document_list.append(new_event)

print(len(document_list))
document_list.sort(key=lambda x: x["idx"])
with open("finance_add.json", "w") as f:
    f.write(json.dumps(document_list))

    # break
