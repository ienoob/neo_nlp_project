#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import gensim
import jieba
import numpy as np
from gensim.models import keyedvectors

# model = keyedvectors.load_word2vec_format("D:\data\word2vec\\light_Tencent_AILab_ChineseEmbedding.bin", binary=True)

data_path = "D:\\data\\文本匹配\\paws-x-zh\\train.tsv"
dev_path = "D:\\data\\文本匹配\\paws-x-zh\\dev.tsv"

with open(data_path, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = f.read()

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
       raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

def load_word_vector(embedding_file):
    embedding_dict = dict()
    with open(embedding_file, encoding="utf-8") as f:
        for line in f:
            if len(line.rstrip().split(" ")) <= 2:
                continue
            token, vector = line.rstrip().split(" ", 1)
            embedding_dict[token] = np.fromstring(vector, dtype=np.float, sep=" ")
    return embedding_dict


class Word2vec(object):

    def __init__(self):
        # self.model = keyedvectors.load_word2vec_format("D:\data\word2vec\\light_Tencent_AILab_ChineseEmbedding.bin", binary=True)
        path = "D:\data\word2vec\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
        # f = open(path, "r", encoding="utf-8")
        self.model = load_word_vector(path)
        self.threshold = 0.5


    def fit(self, input_documents):
        score_list = []

        for data_item in input_documents:
            data_item = data_item.strip()
            if len(data_item) == 0:
                continue
            if len(data_item.split("\t")) == 1:
                # print(data, "hello")
                continue
            sentence1, sentence2, label = data_item.split("\t")
            score = self.predict_one(sentence1, sentence2)
            # print(score)
            score_list.append((score, label))
        threshold_list = [0.1*i for i in range(1, 10)]
        max_acc = 0
        max_ts = -1
        for tl in threshold_list:
            acc = 0
            for s, l in score_list:
                pred_l = 0
                if s > tl:
                    pred_l = 1
                if l == pred_l:
                    acc += 1
            if acc > max_acc:
                max_ts = tl
                max_acc = acc
        self.threshold = max_ts

    def predict_one(self, query, document):
        query_word_embed = [self.model[word] for word in jieba.cut(query) if word in self.model]
        if len(query_word_embed) == 0:
            return 0
        query_word_embed_num = len(query_word_embed)
        q = None
        for qw in query_word_embed:
            ab = np.array([qi for qi in qw])
            if q is None:
                q = ab
            else:
                q += ab
        q /= query_word_embed_num
        document_word_embed = [self.model[word] for word in jieba.cut(document) if word in self.model]
        if len(document_word_embed) == 0:
            return 0
        document_word_embed_num = len(document_word_embed)
        d = None
        for qw in document_word_embed:
            ab = np.array([qi for qi in qw])
            if d is None:
                d = ab
            else:
                d += ab
        d /= document_word_embed_num
        # print(query_word_embed)
        # print(document_word_embed)
        # for word1 in query_word_embed:
        #     for word2 in document_word_embed:
        #         print(cosine_distance(word1, word2))
        # if query_word_embed == 0 or document_word_embed == 0:
        #     return 0

        return cosine_distance(q, d)
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
            print(score)
            pred_value = 0
            if score > self.threshold:
                pred_value = 1
            score_list.append(score)
            if pred_value == int(label):
                acc_num += 1
        print(acc_num/len(input_documents))
        return score_list


model = Word2vec()
train_data_list = train_data.split("\n")
dev_data_list = dev_data.split("\n")
#
model.fit(train_data_list)
print("train complete threshold is {}".format(model.threshold))
model.predict(dev_data_list)
