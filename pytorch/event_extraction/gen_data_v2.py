#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
import pandas as pd
from nlp_applications.ner.crf_model import CRFNerModel
data_path = "D:\data\\tj_event\\trz_events.csv"

data = pd.read_csv(data_path, encoding="gbk")
indx = 27
content = data["content"][indx]
role2id = {'被投资方': 0, '融资金额': 1, '披露时间': 2, '投资方': 3, '融资轮次': 4, '事件时间': 5, '领投方': 6}
id2role = {str(v): k for k, v in role2id.items()}
# print()

model = CRFNerModel()
model.save_model = "D:\Work\git\\neo_nlp_project\pytorch\event_extraction\\finance.model"

model.load_model()
print(content)

text_sentence = re.split("[。\r\n]", content)
input_list = []
for sentence in text_sentence:
    sentence = sentence.strip()
    if sentence == "":
        continue
    extract_res = model.extract_ner(sentence)
    for e_res in extract_res:
        print(id2role[e_res[2]], e_res[3], sentence)
        v = input("input:")
        input_list.append({
            "start": e_res[0],
            "end": e_res[1],
            "span": e_res[2],
            "role": id2role[e_res[2]],
            "content": sentence,
            "if_true": v})

with open("D:\data\\tj_event\data\{}.jsonline", "a") as f:
    f.write(json.dumps({"{}".format(indx): input_list})+"\r\n")

