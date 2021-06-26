#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/26 20:58
    @Author  : jack.li
    @Site    : 
    @File    : evaluation.py

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
    if start is not None:
        extract_ner.append((start, len(input_label), label))
    return extract_ner


def eval_metrix(hit_num, true_num, predict_num):
    recall = (hit_num + 1e-8) / (true_num + 1e-3)
    precision = (hit_num + 1e-8) / (predict_num + 1e-3)
    f1_value = 2 * recall * precision / (recall + precision)

    return {
        "recall": recall,
        "precision": precision,
        "f1_value": f1_value
    }