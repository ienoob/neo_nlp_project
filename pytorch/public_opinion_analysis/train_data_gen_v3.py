#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import json
from data_label import data_label
data_path = "train.json"

with open(data_path, "r") as f:
    data = f.read()

data_list = json.loads(data)
data_label_info = {dt["id"]: dt.get("sentence_idx", []) for dt in data_label}

label2id = {'质押': 1, '企业收购': 2, '中标': 3, '股份回购': 4, '股东减持': 5,
            '企业破产': 6, '股东增持': 7, '企业融资': 8, '亏损': 9, '被约谈': 10,
            '公司上市': 11, '高管变动': 12, '解除质押': 13}
id2label = {v: k for k, v in label2id.items()}
entity_bio_dict = {"O": 0, "PERSON-S": 1, "PERSON-I": 2, "ORG-S": 3, "ORG-I": 4}
pos_neg_event = {
    "质押": 2,
    "企业收购": 0,
    "中标": 1,
    "股份回购": 1,
    "股东减持": 2,
    "企业破产": 2,
    "股东增持": 1,
    "亏损": 2,
    "被约谈": 2,
    "公司上市": 1,
    "高管变动": 0,
    "解除质押": 1,
    "企业融资": 1
}
pos_neg_event_id_version = {
  label2id[k]: v  for k, v in pos_neg_event.items()
}

train_dataset = []
train_dataset_v2 = []
for dt in data_list[:500]:

    # print(dt["data_item"])
    d_id = dt["data"]["id"]
    print(d_id)
    dm = dict()
    for event_item in data_label_info[d_id]:
        event_type = event_item[0]
        if event_type not in {'质押', '企业收购', '中标', '股份回购', '股东减持', '企业破产', '股东增持', '企业融资', '亏损', '被约谈', '公司上市', '高管变动', '解除质押'}:
            raise Exception
        for idx in event_item[1]:
            m = dt["data_item"][idx]
            dm.setdefault(idx, [])
            dm[idx].append(label2id[event_type])


    # print(data_label_info[d_id])

    # for i, it in enumerate(dt["data_item"]):
    #
    #
    #     sentence = it["sentence"]
    #     print(i, sentence)

    with open("D:\data\舆情分析\self-label\{}.txt".format(dt["data"]["id"]), "r", encoding="utf-8") as f:
        ner_label = f.read()
    ner_label = ner_label.split("\n")
    print(len(ner_label))

    batch_sentence = []
    batch_label = []
    sentence = []
    label = []

    for ner in ner_label:
        if ner == "======================":
            # print(sentence)
            batch_sentence.append(sentence)
            sentence = []
            batch_label.append(label)
            label = []
        else:
            sentence.append(ner.split("\t")[0])
            label.append(ner.split("\t")[1])

            if ner.split("\t")[1] not in ["O", "PERSON-S", "PERSON-I", "ORG-S", "ORG-I"]:
                print(ner.split("\t")[1])
                raise Exception
    if sentence:
        batch_sentence.append(sentence)
        batch_label.append(label)

    train_dataset_v2.append({
        "id": d_id,
        "text": dt["data"]["text"],
        "other": dt["data"],
        "batch_sentence": batch_sentence,
        "batch_label": batch_label,
        "data_event": dm
    })



    pos_neg_label = []
    event_label = []
    for i, sent in enumerate(batch_sentence):
        if i not in dm:
            event_label.append([])
            pos_neg_label.append(0)
        else:
            event_label.append(dm[i])
            pn_score = sum([pos_neg_event_id_version[v] for v in dm[i]])
            pn_label = 0
            if pn_score > 0:
                pn_label = 1
            elif pn_score < 0:
                pn_label = 2
            pos_neg_label.append(pn_label)
    # print(event_label)

    # event_sentence = []
    # for event_item in data_label_info[d_id]:
    #     event_type = event_item[0]
    #     if event_type not in {'质押', '企业收购', '中标', '股份回购', '股东减持', '企业破产', '股东增持', '企业融资', '亏损', '被约谈', '公司上市', '高管变动', '解除质押'}:
    #         raise Exception
    #     for idx in event_item[1]:
    #         m = dt["data_item"][idx]
            # print(m)
            # print(m)
    # print(len(batch_sentence))
    # print(len(dt["data_item"]))
    assert len(batch_sentence) == len(dt["data_item"])

    offset = dict()
    for i, it in enumerate(batch_sentence):
        i_sentence = []
        i_ner = []
        for ji, s in enumerate(it):
            if s in {" ", "\n", "\u2003", "\u200b", "\ufeff", "\xa0", "\u3000", "\x06", "\x07", "\u2002"}:
                continue
            i_sentence.append(s)

            i_ner.append(entity_bio_dict[batch_label[i][ji]])
        if len(i_sentence) < 510:

            train_dataset.append((i_sentence, i_ner, event_label[i], pos_neg_label[i]))
        else:
            n_ii_sentence = []
            n_ii_ner = []
            for ii, i_s in enumerate(i_sentence):
                n_ii_sentence.append(i_s)
                n_ii_ner.append(i_ner[ii])
                if i_s == "，":
                    train_dataset.append((n_ii_sentence, n_ii_ner, event_label[i], pos_neg_label[i]))
                    n_ii_sentence = []
                    n_ii_ner = []
            if n_ii_sentence:
                train_dataset.append((n_ii_sentence, n_ii_ner, event_label[i], pos_neg_label[i]))



    # batch_len = [len(sent) for sent in batch_sentence if len(sent)>512]
    # print(batch_len)
    # break

with open("D:\data\self-data\\opinion_analysis_v1.json", "w") as f:
    f.write(json.dumps(train_dataset_v2))

