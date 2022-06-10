#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import copy
import json


# 假的模型解析
def faker_extractor(input_str: str):
    # 返回数据的格式
    return [{'被投资方': ["a"], '融资金额': [1], '披露时间': [2], '投资方': [3], '融资轮次': [4], '事件时间': [5], '领投方': [6]}]


def eval_metrix(hit_num, true_num, predict_num):
    recall = (hit_num + 1e-8) / (true_num + 1e-3)
    precision = (hit_num + 1e-8) / (predict_num + 1e-3)
    f1_value = 2 * recall * precision / (recall + precision)

    return {
        "recall": recall,
        "precision": precision,
        "f1_value": f1_value
    }


# 测试代码
def test_model():
    with open("finance.json", "r") as f:
        datas = f.read()

    datas = json.loads(datas)
    hit_num = 0
    rel_num = 0
    pre_num = 0
    bad_case = []
    role_indicate = dict()
    account_role = {"num": 0}
    for data in datas:

        # faker_extractor 替换成投融资模型
        extractor_event = faker_extractor(data["text"])
        for p_event in extractor_event:
            for k, v in p_event.items():
                role_indicate.setdefault(k, {"hit": 0, "pred": 0, "real": 0})
                role_indicate[k]["pred"] += len(v)

        tp_hit = 0
        for event in data["event"]:
            account_role["num"] += 1
            r_event = dict()
            for r_arg in event["arguments"]:
                r_event.setdefault(r_arg["role"], [])
                r_event[r_arg["role"]].append(r_arg["argument"])

            for k, v in r_event.items():
                role_indicate.setdefault(k, {"hit": 0, "pred": 0, "real": 0})
                role_indicate[k]["real"] += len(v)
                v.sort()
                r_event[k] = v

                account_role.setdefault(k, 0)
                account_role[k] += 1

            max_hit = 0
            max_d = dict()
            for p_event in extractor_event:
                state = 1
                sub_hit = 0
                sub_d = dict()
                for k, v in r_event.items():
                    if k in p_event and p_event[k] == v:
                        sub_hit += 1
                        sub_d[k] = v

                if sub_hit == len(r_event) and sub_hit == len(p_event):
                    hit_num += 1
                    tp_hit += 1
                if sub_hit > max_hit:
                    max_hit = sub_hit
                    max_d = copy.deepcopy(sub_d)
            for k, v in max_d.items():
                role_indicate.setdefault(k, {"hit": 0, "pred": 0, "real": 0})
                role_indicate[k]["hit"] += len(v)

        if tp_hit != len(data["event"]) or tp_hit != len(extractor_event):
            data["predict"] = extractor_event
            bad_case.append(data)

        rel_num += len(data["event"])
        pre_num += len(extractor_event)
        # break

    # 这里展示每个角色的结果
    for role, role_ind in role_indicate.items():
        print("{} : {}".format(role, eval_metrix(role_ind["hit"], role_ind["real"], role_ind["pred"])))
    # print(account_role)
    print(hit_num, rel_num, pre_num)
    # 这里展示以事件为单位的结果
    print(eval_metrix(hit_num, rel_num, pre_num))
    # 这是显示bad case的数量
    print("bad case num {}".format(len(bad_case)))

test_model()
