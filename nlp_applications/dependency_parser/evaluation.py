#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/7/3 10:01
    @Author  : jack.li
    @Site    : 
    @File    : evaluation.py

"""


# unlabeled attachment score
def uas(input_ua, input_ua_):
    precision_value = len(set(input_ua) & set(input_ua_)) / len(input_ua_)
    recall_value = len(set(input_ua) & set(input_ua_)) / len(input_ua)

    return 2*precision_value*recall_value/(precision_value+recall_value)


# labeled attachment score
def las(input_la, input_la_):
    precision_value = len(set(input_la) & set(input_la_)) / len(input_la_)
    recall_value = len(set(input_la) & set(input_la_)) / len(input_la)

    return 2 * precision_value * recall_value / (precision_value + recall_value)
