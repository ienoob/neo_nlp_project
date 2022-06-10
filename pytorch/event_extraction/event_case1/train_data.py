#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json

with open("finance_add.json", "r") as f:
    datas = f.read()



datas = json.loads(datas)
hit_num = 0
rel_num = 0
pre_num = 0
bad_case = []
role_indicate = dict()
account_role = {"num": 0}
print(len(datas))
idx = 1
print(idx)
for data in datas[idx:]:
    # if data["idx"] != 110:
    #     continue
    print("indx ", data["idx"])
    data_id = data["id"]
    # if data_id != "b50d5ea8cc299912b980a82b11b6808e":
    #     continue
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
        if trigger == "":
            trigger = "融资"
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


import colorama
from colorama import init, Fore, Back, Style
init(autoreset=True)
print('\033[1;31;40m''测试')
