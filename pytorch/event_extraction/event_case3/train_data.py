#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
from pytorch.event_extraction.bert_utils import split_cha

with open("bidding.json", "r") as f:
    datas = f.read()
documents = json.loads(datas)

dt = [
    {
        "idx": 4,
        "id": "5be81048929b4b5ff2b357c2e83e2ec0",
        "cg": [
            {'role': '中标公司', 'argument': '浪潮', "index": [0, 1, 2]}
        ]
    },
    {
        "idx": 25,
        "id": "0a1c3ddc87113ced799f29e53f3a161d",
        "cg": [
            {'role': '中标公司', 'argument': '中国医药', "index": [0, 1, 2, 3, 4, 6]}
        ]
    },
    {
        "idx": 50,
        "id": "0bc0d311096c6c6fceef764d43b4fb66",
        "cg": [
            {'role': '中标公司', 'argument': '杭州园林', "index": [0, 1]}
        ]
    },
    {
        "idx": 51,
        "id": "b0e801c5114abad66876c2b2edb1c6cc",
        "cg": [
            {'role': '中标公司', 'argument': '河北建设集团', "index": [0]}
        ]
    }
]
dt_dict = {klist["id"]: klist for klist in dt}

def get_train_data():
    role2id = {'中标公司': 0, '中标日期': 1, '中标标的': 2, '中标金额': 3, '披露日期': 4, '招标方': 5}
    label2id = {'O': 0, 'B-0': 1, 'I-0': 2, 'B-1': 3, 'I-1': 4, 'B-2': 5, 'I-2': 6, 'B-3': 7, 'I-3': 8, 'B-4': 9, 'I-4': 10, 'B-5': 11, 'I-5': 12}
    role2id_v2 = {'unk': 0}
    for r, rid in role2id.items():
        role2id_v2[r] = rid + 1


    dev_data_size = 260

    rt_data = {
        "bert_data": {
            "train": [],
            "dev": [],
            "all": []
        },
        "crf_data": {
            "train": [],
            "dev": [],
            "all": []
        }
    }

    for ii, doc in enumerate(documents):

        if doc.get("title"):
             text = doc["title"] + "\n" + doc["text"]
        else:
            text = doc["text"]
        event_list = doc["event"]

        label_data = ["O" for _ in text]
        label_entity = []
        entity_num = 0

        for sub_event in event_list:
            sub_event["arguments"].sort(key=lambda x: x["role"])
            for arg in sub_event["arguments"]:

                if arg["role"] not in role2id:
                    role2id[arg["role"]] = len(role2id)
                    label2id["B-{}".format(role2id[arg["role"]])] = len(label2id)
                    label2id["I-{}".format(role2id[arg["role"]])] = len(label2id)

                argument = arg["argument"]\
                    .replace("*", "\*")\
                    .replace("+", "\+")\
                    .replace("[", "\[")\
                    .replace("-", "\-")\
                    .replace("]", "\]")\
                    .replace(")", "\)")\
                    .replace("(", "\(")
                # print(argument)
                res = re.finditer(argument, text, re.M)
                res_list = [(rs.group(), rs.span()) for rs in res]

                if doc["id"] in dt_dict:
                    for kl in dt_dict[doc["id"]]["cg"]:
                        if kl["role"] == arg["role"] and kl["argument"] == arg["argument"]:
                            res_list = [res_list[sub_index] for sub_index in kl["index"]]

                            break
                for rs in res_list:
                    assert rs[0] == arg["argument"]
                    start, end = rs[1]
                    label_data[start] = "B-{}".format(role2id[arg["role"]])

                    for iv in range(start + 1, end):
                        label_data[iv] = "I-{}".format(role2id[arg["role"]])

                    label_entity.append((start, end, role2id_v2[arg["role"]]))
                    # label_entity.append((start, end, role2id[arg["role"]]))

        label_list_id = [label2id[lb] for lb in label_data]
        start = 0
        for iiv, tt in enumerate(text):
            if tt == "\n":
                if iiv > start:
                    sub_data = {
                        "text": text[start:iiv],
                        "label": label_data[start:iiv],
                        "entity": label_entity
                    }
                    rt_data["crf_data"]["all"].append(sub_data)
                    if ii >= dev_data_size:
                        rt_data["crf_data"]["train"].append(sub_data)
                    else:
                        rt_data["crf_data"]["dev"].append(sub_data)
                start = iiv + 1

        if len(text) < 500:

            assert len(text) == len(label_list_id)
            split_word = []
            split_label = []
            for iv, t in enumerate(text):
                if t in split_cha:
                    continue
                split_word.append(t)
                split_label.append(label_list_id[iv])
            sub_data = {
                "text": split_word,
                "label": split_label,
                "entity": label_entity
            }
            rt_data["bert_data"]["all"].append(sub_data)
            if ii >= dev_data_size:
                rt_data["bert_data"]["train"].append(sub_data)
            else:
                rt_data["bert_data"]["dev"].append(sub_data)

        else:
            text_sentence = text.split("\n")
            text_cut = []
            text_cache = []
            cache_len = 0

            for sentence in text_sentence:
                if len(sentence) > 500:
                    continue
                if cache_len + len(text_cache) - 1 + len(sentence) > 500:
                    text_cut.append("\n".join(text_cache))
                    cache_len = 0
                    text_cache = []

                text_cache.append(sentence)
                cache_len += len(sentence)

            # print(len(text))
            if text_cache:
                text_cut.append("\n".join(text_cache))
            if len(text_cut) == 1:
                continue

            last_indx = 0
            for sub_text in text_cut:
                # print(len(sub_text))
                sub_label_list = label_list_id[last_indx:last_indx + len(sub_text)]
                assert len(sub_text) == len(sub_label_list)
                split_word = []
                split_label = []
                for iv, t in enumerate(sub_text):
                    if t in split_cha:
                        continue
                    split_word.append(t)
                    split_label.append(sub_label_list[iv])

                sub_data = {
                    "text": split_word,
                    "label": split_label,
                    "entity": label_entity
                }
                rt_data["bert_data"]["all"].append(sub_data)
                if ii >= dev_data_size:
                    rt_data["bert_data"]["train"].append(sub_data)
                else:
                    rt_data["bert_data"]["dev"].append(sub_data)

                # rs = extract_entity(label_data[last_indx:last_indx+len(sub_text)])
                # for rsi in rs:
                #     print(sub_text[rsi[0]:rsi[1]])
                last_indx += len(sub_text) + 1

        entity_num += len(label_entity)
    rt_data["label2id"] = label2id
    rt_data["role2id"] = role2id
    rt_data["role2id_v2"] = role2id_v2
    return rt_data

rt_data = get_train_data()

print("document {}".format(len(documents)))
print("valid train {}".format(len(rt_data["bert_data"]["all"])))
print("crf train {}".format(len(rt_data["crf_data"]["all"])))
# print("entity num {}".format(entity_num))
print("label2id {}".format(rt_data["label2id"]))
print("role2id {}".format(rt_data["role2id"]))
