#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/2/20 20:52
    @Author  : jack.li
    @Site    : 
    @File    : evaluation.py
    增加评估ner 结果模型，使用严格策略，
    即位置和分类都对才能算抽取准确
"""


def extract_entity(input_label):
    start = None
    label = None
    extract_ner = []
    for i, x in enumerate(input_label):
        if x == "O":
            if start is not None:
                extract_ner.append((start, i, label))
                start = None
                label = None
        else:
            xindex, xlabel = x.split("-")
            if xindex == "B":
                if start is not None:
                    extract_ner.append((start, i, label))
                start = i
                label = xlabel
            else:
                if label != xlabel:
                    start = None
                    label = None
    return extract_ner


# 这里输入的BIO类型的数据
def metrix(true_labels, predict_labels):
    true_res = 0
    pred_res = 0
    predict_true = 0

    assert len(true_labels) == len(predict_labels)

    for i, label in enumerate(true_labels):
        pred_label = predict_labels[i]

        assert len(label) == len(pred_label)

        true_entity = extract_entity(label)
        pred_entity = extract_entity(pred_label)

        true_res += len(true_entity)
        pred_res += len(pred_entity)

        d_true = {(s, e): lb for s, e, lb in true_entity}
        d_pred = {(s, e): lb for s, e, lb in pred_entity}

        for k, v in d_true.items():
            if k in d_pred and d_pred[k] == v:
                predict_true += 1

    recall = predict_true*1.0/true_res
    precision = predict_true*1.0/pred_res

    return recall, precision


