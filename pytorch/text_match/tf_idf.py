#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
import jieba
import pandas

data_path = "D:\\data\\文本匹配\\paws-x-zh\\train.tsv"
dev_path = "D:\\data\\文本匹配\\paws-x-zh\\dev.tsv"

with open(data_path, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = f.read()

# df = pandas.read_csv(data_path, sep='\t')
#

# print(df.head())
class TFModel(object):

    def __init__(self):
        pass

    def fit(self, input_data_list):
        predict_list = []
        label_list = []
        for data_item in input_data_list:
            data_item = data_item.strip()
            if len(data_item) == 0:
                continue
            if len(data_item.split("\t")) == 1:
                # print(data, "hello")
                continue
            sentence1, sentence2, label = data_item.split("\t")
            d1 = dict()
            for word in jieba.cut(sentence1):
                d1.setdefault(word, 0)
                d1[word] += 1

            d2 = dict()
            for word in jieba.cut(sentence2):
                d2.setdefault(word, 0)
                d2[word] += 1

            d1_norm = 0.0
            d12_trunc = 0.0
            for k, v in d1.items():
                d1_norm += v ** 2
                if k in d2:
                    d12_trunc += v * d2[k]

            d1_norm = np.sqrt(d1_norm)

            d2_norm = 0.0
            for k, v in d2.items():
                d2_norm += v ** 2

            predict_list.append(d12_trunc / d1_norm / d2_norm)
            label_list.append(int(label))
        acc_num = 0
        for i, p in enumerate(predict_list):
            pred_label = 1 if p > 0.5 else 0
            if pred_label == label_list[i]:
                acc_num += 1

        acc_rate = acc_num / len(predict_list)

        return acc_rate






