#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/25 21:03
    @Author  : jack.li
    @Site    : 
    @File    : pattern_model.py

"""
"""
    这个方法名为模板方法，但是模板是从数据中来，和人工指定模板的思路不同
"""
import re
import json
from nlp_applications.data_loader import load_json_line_data

schema_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取\duee_schema\\duee_event_schema.json"
data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取\duee_train.json\\duee_train.json"


schema_data = load_json_line_data(schema_path)
train_data = load_json_line_data(data_path)

for schema in schema_data:
    print(schema)


def sample_data(event_type, event_role):
    data_list = []
    for data in train_data:
        for event in data["event_list"]:
            if event["event_type"] != event_type:
                continue
            for arg in event["arguments"]:
                if arg["role"] != event_role:
                    continue
                data_list.append((data["text"], arg["argument_start_index"], arg["argument"]))

    return data_list


class PatternModel(object):

    def __init__(self):
        self.pattern_list = []


    def fit(self, input_feature_data, label_datas):
        pattern_statis = dict()
        for i, text in enumerate(input_feature_data):
            label_data = label_datas[i]
            start_indx = label_data[0]
            end_indx = label_data[0]+len(label_data[1])
            start_context = text[:start_indx]
            core_data = label_data[1]
            end_context = text[end_indx:]

            start_p = start_context[-1] if len(start_context) else "$"
            end_p = end_context[0] if len(end_context) else "$"

            pattern_statis.setdefault((start_p, end_p), 0)
            pattern_statis[(start_p, end_p)] += 1
        pattern_list = [(p[0], p[1], c) for p, c in pattern_statis.items()]
        pattern_list.sort()
        pattern_list_value = []
        last = None
        for p1, p2, c in pattern_list:
            if last is not None:
                if last[0] != p1:
                    pattern_list_value.append(last)
                    last = (p1, [p2], c)
                else:
                    last = (p1, [p2]+last[1], c+last[2])
            else:
                last = (p1, [p2], c)
        pattern_list_value.append(last)
        pattern_list_value.sort(key=lambda x: x[2], reverse=True)

        for p1, p2, _ in (pattern_list_value):
            pattern = r"{0}(.+?){1}".format(p1, p2)
            self.pattern_list.append(pattern)

    def predict(self, input_text):
        predict_res = []
        for text in input_text:
            extract = []
            for pt in self.pattern_list:
                g = re.search(pt, text)
                if g:
                    extract.append((g.start(1),g.group(1)))
                    break
            predict_res.append(extract)

        return predict_res



test_train = sample_data("灾害/意外-爆炸", "时间")
test_train_feature = [d[0] for d in test_train]
test_train_label = [(d[1],d[2]) for d in test_train]

pt = PatternModel()
pt.fit(test_train_feature, test_train_label)
pres = pt.predict(test_train_feature)
print(test_train_label)
print(pres)