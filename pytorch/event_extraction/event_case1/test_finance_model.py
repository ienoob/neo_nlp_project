#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import copy
import json
from nlp_applications.ner.evaluation import eval_metrix, extract_entity

from pytorch.event_extraction.event_case1.crf_model import crf_extractor
from pytorch.event_extraction.event_case1.bert_extractor import bert2extract_ner


if __name__ == "__main__":

    with open("finance.json", "r") as f:
        datas = f.read()
    # with open("finance_add.json", "r") as f:
    #     data_add = f.read()
    #
    # add_documents = json.loads(data_add)[:15]

    datas = json.loads(datas)

    hit_num = 0
    rel_num = 0
    pre_num = 0
    bad_case = []
    role_indicate = dict()
    account_role = {"num": 0}
    for data in datas:
        extractor_event = bert2extract_ner(data["text"])
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

                # if len(r_event) != len(p_event):
                #     continue
                # state = 1
                # for k, v in r_event.items():
                #     if k not in p_event:
                #         state = 0
                #         break
                #     v.sort()
                #     if v != p_event[k]:
                #         state = 0
                #         break
                # if state:
                #     hit_num += 1
                #     tp_hit += 1
        if tp_hit != len(data["event"]) or tp_hit != len(extractor_event):
            data["predict"] = extractor_event
            bad_case.append(data)

        rel_num += len(data["event"])
        pre_num += len(extractor_event)
        # break

    # for role, role_ind in role_indicate.items():
    #     print("{} : {}".format(role, eval_metrix(role_ind["hit"], role_ind["real"], role_ind["pred"])))
    # print(account_role)
    print(hit_num, rel_num, pre_num)
    print(eval_metrix(hit_num, rel_num, pre_num))
    print("bad case num {}".format(len(bad_case)))
    for b_data in bad_case[0:]:
        print(b_data["title"])
        print(b_data["text"])
        print(b_data["event"])
        print(b_data["predict"])
        print(b_data["id"])

        break

    # with open("base_case.json", "w") as f:
    #     f.write(json.dumps(bad_case))
