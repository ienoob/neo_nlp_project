#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json

with open("bidding.json", "r") as f:
    datas = f.read()


datas = json.loads(datas)
hit_num = 0
rel_num = 0
pre_num = 0
bad_case = []
role_indicate = dict()
account_role = {"num": 0}
print(len(datas))
idx = 63
print(idx)
for data in datas[idx:]:
    data_id = data["id"]
    print(data_id)

    if data.get("title"):
        text = data["title"] + "\n" + data["text"]
    else:
        text = data["text"]
    print(text)
    print(data["event"])
    print(len(text))

    event_list = data["event"]
    print(len(event_list))
    for sub_event in event_list:
        trigger = sub_event["trigger"]
        res = re.finditer(trigger, text)
        for rs in res:
            print("trigger ", rs.span(), rs.group())
        # sub_event["arguments"].sort(key=lambda x: x["role"])

        for arg in sub_event["arguments"]:
            print(arg)
            argument = arg["argument"].replace("*", "\*").replace("+", "\+").replace("(", "\(").replace(")", "\)")

            res = re.finditer(argument, text, re.M)
            for rs in res:
                print(rs.span(), rs.group())

    break
