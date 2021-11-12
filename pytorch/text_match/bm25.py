#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import jieba
import numpy as np
data_path = "D:\\data\\文本匹配\\paws-x-zh\\train.tsv"
dev_path = "D:\\data\\文本匹配\\paws-x-zh\\dev.tsv"

with open(data_path, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = f.read()


class BM25Model(object):

    def __init__(self):
        self.idf_dict = dict()
        self.k1 = 1
        self.b = 0.5
        self.avg_len = 0

    def fit(self, input_documents):
        for data_item in input_documents:
            data_item = data_item.strip()
            if len(data_item) == 0:
                continue
            if len(data_item.split("\t")) == 1:
                # print(data, "hello")
                continue
            sentence1, sentence2, label = data_item.split("\t")
            word_set = set()
            for word in jieba.cut(sentence2):
                word_set.add(word)
                self.avg_len += 1
            for word in word_set:
                self.idf_dict.setdefault(word, 0)
                self.idf_dict[word] += 1
        big_n = len(input_documents)
        self.avg_len /= big_n
        for k, v in self.idf_dict.items():
            self.idf_dict[k] = np.log((big_n-v+0.5)/(v+0.5)+1)

    def predict_one(self, query, input_document):
        tf_dict = dict()
        doc_len = 0
        for word in jieba.cut(input_document):
            tf_dict.setdefault(word, 0)
            tf_dict[word] += 1
            doc_len += 1
        big_k = self.k1*(1-self.b+self.b*doc_len/self.avg_len)
        score = 0.0
        for word in jieba.cut(query):
            idf_value = self.idf_dict.get(word, 0.001)
            r_value = (self.k1+1)*tf_dict.get(word, 0.001)/(big_k+tf_dict.get(word, 0.001))
            score += idf_value*r_value

        return score
    def predict(self, input_documents):
        score_list = []

        acc_num = 0
        for data_item in input_documents:
            data_item = data_item.strip()
            if len(data_item) == 0:
                continue
            if len(data_item.split("\t")) == 1:
                # print(data, "hello")
                continue
            sentence1, sentence2, label = data_item.split("\t")
            score = self.predict_one(sentence1, sentence2)
            pred_value = 0
            if score > 90:
                pred_value = 1
            score_list.append(score)
            if pred_value == int(label):
                acc_num += 1
        print(acc_num/len(input_documents))
        return score_list


train_data_list = train_data.split("\n")
dev_data_list = dev_data.split("\n")

bm_model = BM25Model()
bm_model.fit(train_data_list)

score_res = bm_model.predict(dev_data_list)
print(score_res[:20])

