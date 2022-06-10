#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
from pytorch.event_extraction.event_case3.train_data import rt_data

trigger_set = {'招标', '连中三标', '中选', '中标', '承建'}
role2id = rt_data["role2id"]
label2id = rt_data["label2id"]


def split(input_str):
    split_char = {"。", "\n", "\r"}
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


# def train():
#     with open("bidding.json", "r") as f:
#         data = f.read()
#
#     documents = []
#     data = json.loads(data)
#
#     train_documents = []
#     dev_documents = []
#     for i, sub_train_data in enumerate(data):
#         text = sub_train_data["text"]
#         title = sub_train_data["title"]
#         doc_id = sub_train_data["id"]
#         event_list = sub_train_data["event"]
#
#         text_line = list(split(text)) + [title]
#         for line in text_line:
#             line = line.strip()
#             if line == "":
#                 continue
#             state = 0
#             for trigger_word in trigger_set:
#                 if trigger_word in line:
#                     state = 1
#                     break
#             if state == 0:
#                 continue
#
#             label_list = ["O" for _ in line]
#             location = []
#             c_state = 0
#             for sub_event in event_list:
#                 # trigger = sub_event["trigger"]
#                 sub_event["arguments"].sort(key=lambda x: x["role"])
#                 for arg in sub_event["arguments"]:
#                     if arg["role"] not in role2id:
#                         role2id[arg["role"]] = len(role2id)
#                         label2id["B-{}".format(role2id[arg["role"]])] = len(label2id)
#                         label2id["I-{}".format(role2id[arg["role"]])] = len(label2id)
#                     if arg["argument"] in line:
#                         index = line.index(arg["argument"])
#                         label_list[index] = "B-{}".format(role2id[arg["role"]])
#                         for idx in range(index + 1, len(arg["argument"]) + index):
#                             label_list[idx] = "I-{}".format(role2id[arg["role"]])
#                         location.append((role2id[arg["role"]], index, index + len(arg["argument"])))
#                         c_state = 1
#             if c_state == 0:
#                 continue
#             # print(label_list)
#             label_list_id = [label2id[lb] for lb in label_list]
#
#             if i < 260:
#                 dev_documents.append((line, label_list_id, label_list))
#             else:
#                 train_documents.append((line, label_list_id, label_list))
#                 # documents.append((line, label_list_id, label_list))
#     return train_documents, dev_documents


# train_data_lists, dev_data_lists = train()
train_data_lists = rt_data["crf_data"]["train"]
dev_data_lists = rt_data["crf_data"]["dev"]

role2id = rt_data["role2id"]
id2role = {v:k for k, v in role2id.items()}


def crf_model():
    from nlp_applications.ner.crf_model import CRFNerModel, sent2features,  sent2labels
    from nlp_applications.ner.evaluation import metrix_v2, extract_entity, eval_metrix

    crf_mode = CRFNerModel()
    crf_mode.save_model = "bidding.model"
    X_train = [sent2features(s["text"]) for s in train_data_lists]
    # print(X_train[0])
    y_train = [sent2labels(s["label"]) for s in train_data_lists]

    X_dev = [sent2features(s["text"]) for s in dev_data_lists]
    # print(X_train[0])
    y_dev = [sent2labels(s["label"]) for s in dev_data_lists]
    # print(y_train[0])

    crf_mode.fit(X_train, y_train)

    predict_labels = crf_mode.predict_list(X_dev)

    role_indicate = dict()
    for i, pred in enumerate(predict_labels):
        real_row = y_dev[i]

        true_entity = extract_entity(real_row)
        pred_entity = extract_entity(pred)

        for e in true_entity:
            role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
            role_indicate[e[2]]["real"] += 1

        for e in pred_entity:
            role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
            role_indicate[e[2]]["pred"] += 1

        for e in true_entity:
            if e in pred_entity:
                role_indicate[e[2]]["hit"] += 1

    for role_id, role_ind in role_indicate.items():
        print("{} : {}".format(id2role[int(role_id)], eval_metrix(role_ind["hit"], role_ind["real"], role_ind["pred"])))

    print(metrix_v2(y_dev, predict_labels))
    print(predict_labels[0])

    crf_mode.dump_model()
print(role2id)
print(label2id)
crf_model()
