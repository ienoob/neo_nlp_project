#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json

with open("D:\data\self-data\\alias_train_v1.json", "r", encoding="utf-8") as f:
    data = f.read()
train_list = json.loads(data)
n_train_list = []
for i, dt in enumerate(train_list):
    print(i)
    if i in [55, 98, 99, 100, 119, 130, 135, 188, 191, 215, 226, 245, 253, 268, 270, 307, 315, 335, 336, 344, 366, 325]:
        continue
    # print(dt["orient_sentence"])
    # print(dt["full_short_list"])
    print(dt["golden_answer"])
    offset = dt["offset"]
    offset_rev = {v: int(k) for k, v in offset.items()}
    # print(offset_rev)
    for ans in dt["full_short_list"]:
        f_s, f_e, s_s, s_e = ans["key"]
        f_s = offset_rev[f_s]
        f_e = offset_rev[f_e-1]+1
        s_s = offset_rev[s_s]
        s_e = offset_rev[s_e-1]+1

        # print(dt["orient_sentence"][f_s:f_e], dt["orient_sentence"][s_s:s_e])
    n_train_list.append(dt)
    # break

with open("D:\data\self-data\\alias_train_v2.json", "r", encoding="utf-8") as f:
    data = f.read()

filter_char = {" ", "\xa0"}
for dt in json.loads(data):
    iv = 0
    jv = 0
    dv = dict()
    print(dt["sentence"])
    for s in dt["sentence"]:
        if s not in filter_char:
            dv[jv] = iv
            iv += 1
        jv += 1
    offset_rev = {v: int(k) for k, v in dv.items()}
    if len(dt["alias"])==0:
        continue
    print(dt["alias"])
    dt_res = {
        "orient_sentence": dt["sentence"],
        "sentence": [s for s in dt["sentence"] if s not in filter_char],
        "full_short_list": [{"key": (dv[item["name_idx"][0]],
                                     dv[item["name_idx"][1]-1]+1,
                                     dv[item["alias_idx"][0]],
                                     dv[item["alias_idx"][1]-1]+1)}
                            for item in dt["alias"]],
        "golden_answer": [(item["name"], item["alias"]) for item in dt["alias"]]
    }
    print(dt_res["golden_answer"])
    # for item in dt_res["full_short_list"]:
    #     f_s, f_e, s_s, s_e = item["key"]
    #     f_s = offset_rev[f_s]
    #     f_e = offset_rev[f_e - 1] + 1
    #     s_s = offset_rev[s_s]
    #     s_e = offset_rev[s_e - 1] + 1
    #
    #     print(dt_res["orient_sentence"][f_s:f_e], dt_res["orient_sentence"][s_s:s_e])
    for item in dt["alias"]:
        print(item["name_idx"])
        f_s, f_e, s_s, s_e = item["name_idx"][0], item["name_idx"][1], item["alias_idx"][0], item["alias_idx"][1]
        print(dt["sentence"][f_s:f_e], dt["sentence"][s_s:s_e])
    # print(dt_res["full_short_list"])
    # print(dt_res["golden_answer"])
    n_train_list.append(dt_res)

print(len(n_train_list))
train_list = n_train_list
