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

schema_path = "D:\data\句子级事件抽取\duee_schema\\duee_event_schema.json"
data_path = "D:\data\句子级事件抽取\duee_train.json\\duee_train.json"
eval_data_path = "D:\data\句子级事件抽取\duee_dev.json\\duee_dev.json"


schema_data = load_json_line_data(schema_path)
train_data = load_json_line_data(data_path)
eval_data = load_json_line_data(eval_data_path)

train_data = list(train_data)

# for schema in schema_data:
#     print(schema)


# 严格结果， 要求完成重合
def hard_score_res(real_data, predict_data):
    assert len(real_data) == len(predict_data)

    score = 0.0
    d_count = len(real_data)
    p_count = 0
    for i, rd in enumerate(real_data):
        if len(predict_data[i]) is 0:
            continue
        p_count += 1
        if rd[0] == predict_data[i][0] and rd[1] == predict_data[i][1]:
            score += 1

    matrix = {
        "score": score,
        "p_count": p_count,
        "d_count": d_count,
        "recall": (score+1e-8)/(d_count+1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }

    return matrix


# 软评估结果, 不要求完全重合
def soft_score_res(real_data, predict_data):
    assert len(real_data) == len(predict_data)
    score = 0.0
    d_count = len(real_data)
    p_count = 0
    for i, rd in enumerate(real_data):
        pre_res = predict_data[i]
        if len(pre_res) is 0:
            continue
        p_count += 1
        r_s_ind = rd[0]
        r_e_ind = rd[0]+len(rd[1])
        p_s_ind = pre_res[0]
        p_e_ind = pre_res[0]+len(pre_res[1])
        if r_s_ind > p_s_ind or r_e_ind < p_e_ind:
            mix_area = 0.0
        else:
            mix_area = min(r_e_ind, p_e_ind)-max(r_s_ind, p_s_ind)

        all_area = max(r_e_ind, p_e_ind)-min(r_s_ind, p_s_ind)

        score += mix_area/all_area
    matrix = {
        "score": score,
        "p_count": p_count,
        "d_count": d_count,
        "recall": (score + 1e-8) / (d_count + 1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }

    return matrix


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


def bm25(query, doc_list):

    da = []
    d = dict()
    for doc in doc_list:
        di = dict()
        for char in doc:
            d.setdefault(char, 0)
            d[char] += 1
            di.setdefault(char, 0)
            di[char] += 1
        da.append(di)


# Levenshtein
def distance():
    pass


def dtw():
    pass


def max_sub_sequence(seq1, seq2):
    seq1_len = len(seq1)
    seq2_len = len(seq2)
    dp = [[0 for i in range(seq2_len+1)] for j in range(seq1_len+1)]

    for i, s1 in enumerate(seq1):
        for j, s2 in enumerate(seq2):
            if s1 == s2:
                dp[i+1][j+1] = max([dp[i][j]+1, dp[i][j+1], dp[i+1][j]])
            else:
                dp[i+1][j+1] = max([dp[i][j], dp[i][j+1], dp[i+1][j]])

    return dp[-1][-1]


print(max_sub_sequence("abc", "yagbghachje"))




charater_list = [".", "+"]


def single_pattern(input_str):
    pattern = []
    for char in input_str:
        if char.isdigit():
            pattern.append("d")
        elif char.isalpha():
            pattern.append("w")
        elif '\u4e00' <= char <= '\u9fff':
            pattern.append("c")
        else:
            pattern.append(char)

    return pattern


class PatternModel(object):

    def __init__(self):
        self.pattern_list = []
        self.core_list = []
        self.core_pattern = []

    def fit(self, input_feature_data, label_datas):
        pattern_statis = dict()
        for i, text in enumerate(input_feature_data):
            label_data = label_datas[i]
            start_indx = label_data[0]
            end_indx = label_data[0]+len(label_data[1])
            start_context = text[:start_indx]
            core_data = label_data[1]
            self.core_list.append(core_data)
            end_context = text[end_indx:]

            start_p = start_context[-1] if len(start_context) else "$"
            end_p = end_context[0] if len(end_context) else "$"

            pattern_statis.setdefault((start_p, end_p), 0)
            pattern_statis[(start_p, end_p)] += 1

        pattern_list = [(p[0], p[1], c) for p, c in pattern_statis.items()]
        pattern_list.sort()
        pattern_list_value = []
        last = None
        filter_char = [")", "*", "+", "?", "(", "-"]
        for p1, p2, c in pattern_list:
            if p1 in filter_char:
                p1 = "\\{}".format(p1)
            if p2 in filter_char:
                p2 = "\\{}".format(p2)
            if last is not None:
                if last[0] != p1:
                    pattern_list_value.append(last)
                    last = (p1, [p2], c)
                else:
                    last = (p1, [p2]+last[1], c+last[2])
            else:
                last = (p1, [p2], c)
        if last:
            pattern_list_value.append(last)
        pattern_list_value.sort(key=lambda x: x[2], reverse=True)

        for p1, p2, _ in (pattern_list_value):
            pattern = r"{0}(.+?)[{1}]".format(p1, "".join(p2))
            self.pattern_list.append(pattern)

        self.core_pattern = [single_pattern(span) for span in self.core_list]

    def calculate(self, input_str):
        score = 0.0
        for core in self.core_list:
            score += max_sub_sequence(input_str, core)

        return score

    # def calculate_pattern_score(self, input_str):
    #     score = 0.0
    #     for c_pattern in self.core_pattern:
    #         if re.fullmatch(c_pattern, )

    def predict(self, input_text):
        predict_res = []
        for text in input_text:
            extract = tuple()
            extract_infos = []
            for pt in self.pattern_list:
                try:
                    g = re.search(pt, text)
                    if g:
                        e_span = g.group(1)
                        e_span_pattern = single_pattern(e_span)

                        extract_infos.append((g.start(1), e_span, self.calculate(e_span)))
                except Exception as e:
                    print(text, pt)
                    raise Exception
            extract_infos.sort(key=lambda x: x[2], reverse=True)
            if extract_infos:
                extract = (extract_infos[0][0], extract_infos[0][1])
            predict_res.append(extract)

        return predict_res


hit_count = 0.0
pred_count = 0.0
real_count = 0.0
for schema in schema_data:
    event_type = schema["event_type"]
    for role in schema["role_list"]:
        role_value = role["role"]
        test_train = sample_data(event_type, role_value)
        if len(test_train) == 0:
            continue
        test_train_feature = [d[0] for d in test_train]
        test_train_label = [(d[1],d[2]) for d in test_train]

        pt = PatternModel()
        pt.fit(test_train_feature, test_train_label)
        pres = pt.predict(test_train_feature)

        hard_eval = hard_score_res(test_train_label, pres)
        soft_eval = soft_score_res(test_train_label, pres)
        print(soft_eval)

        hit_count += hard_eval["score"]
        pred_count += soft_eval["p_count"]
        real_count += soft_eval["d_count"]



print(hit_count/pred_count, hit_count/real_count)
