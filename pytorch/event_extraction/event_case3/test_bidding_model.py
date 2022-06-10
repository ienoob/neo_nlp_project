#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import copy
import json
from nlp_applications.ner.crf_model import CRFNerModel
from nlp_applications.ner.evaluation import eval_metrix, extract_entity
from pytorch.event_extraction.event_case3.train_data import rt_data
from pytorch.event_extraction.event_case3.bert_extractor import bert2extract_ner

model_path = "bidding.model"

model = CRFNerModel()
model.save_model = model_path
model.load_model()
role2id = rt_data["role2id"]
trigger_set = {'招标', '连中三标', '中选', '中标', '承建'}
id2role = {v: k for k, v in role2id.items()}

label2id = rt_data["label2id"]
id2label = {v: k for k, v in label2id.items()}


class DSU(object):
    def __init__(self, N):
        self.root = [i for i in range(N)]
        self.depth = [1 for i in range(N)]

    def find(self, k):
        if self.root[k] == k:
            return k
        return self.find(self.root[k])

    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        xh = self.depth[x]
        yh = self.depth[y]
        if x == y:
            return
        if xh >= yh:
            self.root[y] = x
            self.depth[x] = max(self.depth[x], self.depth[y] + 1)
        else:
            self.root[x] = y

def split(input_str):
    split_char = {"。", "\n", "\r", "；"}
    not_add_char = {"\r", "\n"}
    start = 0
    for i, i_char in enumerate(input_str):
        if i_char not in split_char:
            continue
        if i > start:
            if i_char in not_add_char:
                sub_str = input_str[start:i]
            else:
                sub_str = input_str[start:i+1]
            sub_str = sub_str.strip()
            yield sub_str
        start = i+1
    if start<len(input_str):
        sub_str = input_str[start:]
        yield sub_str



def merge_event(event_list):
    final_res = []
    event_list.sort(key=lambda x: len(x))

    for i, sub_res in enumerate(event_list):
        state = 1
        for sub_j_res in event_list[i+1:]:
            s = 1
            for k, v in sub_res.items():
                if k not in sub_j_res:
                    s = 0
                    break
                if v != sub_j_res[k]:
                    sub_v_num = 0
                    for sub_v in v:
                        if sub_v in sub_j_res[k]:
                            sub_v_num += 1
                    if sub_v_num != len(v):
                        s = 0
                        break
            if s:
                state = 0
                break
        if state:
            final_res.append(sub_res)

    pair_list = []
    dsu = DSU(len(final_res))
    for i, event_i in enumerate(final_res):
        for j, event_j in enumerate(final_res):
            if j <= i:
                continue
            if event_i.get("中标标的", "a") == event_j.get("中标标的", "b"):
                pair_list.append((i, j))
                dsu.union(i, j)
            # else:


    cluster_list = []
    for i in range(len(final_res)):
        if len(cluster_list) == 0:
            cluster_list.append([i])
        else:
            state = 1
            for j_cluster in cluster_list:
                if dsu.find(i) == dsu.find(j_cluster[0]):
                    j_cluster.append(i)
                    state = 0
                    break
            if state:
                cluster_list.append([i])

    final_event_list = []
    for cluster in cluster_list:
        event = {}
        for idx in cluster:
            for k, v in final_res[idx].items():
                event[k] = v
        if len(event) < 2:
            continue
        final_event_list.append(event)

    return final_event_list


def extractor(input_text):
    # event_res = {}
    final_res = []
    tempt_res = []
    # text_sentence = re.split("[。\r\n]", input_text)
    text_sentence = split(input_text)
    for i, sentence in enumerate(text_sentence):
        t_state = 0
        for t_word in trigger_set:
            if t_word in sentence:
                t_state = 1
                break
        if t_state == 0:
            continue
        event_res = {}
        extract_res = model.extract_ner(sentence)
        for e_res in extract_res:
            # event_res.setdefault(e_res[2], set())
            key = id2role[int(e_res[2])]
            event_res.setdefault(key, [])
            if e_res[3] not in event_res[key]:
                event_res[key].append(e_res[3])
            # event_res[key] = e_res[3]
        # if len(event_res) < 2:
        #     continue
        for k, v in event_res.items():
            v.sort()
            event_res[k] = v
        # merge event
        # for res in event_res:

        tempt_res.append(event_res)

    # merge stage
    final_res = merge_event(tempt_res)
    # for k, v in event_res.items():
    #     final_res[id2role[int(k)]] = list(v)[0]

    return final_res





with open("bidding.json", "r") as f:
    datas = f.read()

datas = json.loads(datas)
hit_num = 0
rel_num = 0
pre_num = 0
bad_case = []
role_indicate = dict()
account_role = {"num": 0}
for data in datas:
    # print(data["content"])
    # print(data)

    # bert2extract_ner(data["text"])
    # break

    # if data["id"] != "96fc6aefbd535b06f90604428dbc8804":
    #     continue
    # print(data["event"])
    # print(data["title"])
    # extractor_event = extractor(data["text"])
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
print(account_role)
print(hit_num, rel_num, pre_num)
print(eval_metrix(hit_num, rel_num, pre_num))
print("bad case num {}".format(len(bad_case)))
for b_data in bad_case[110:]:
    print(b_data["title"])
    print(b_data["text"])
    print(b_data["event"])
    print(b_data["predict"])
    print(b_data["id"])

    break

with open("badcase.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(bad_case))
