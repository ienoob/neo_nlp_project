#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import gensim
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

data_path = "D:\data\depency_parser\evsam05\依存分析训练数据\THU"
train_data_path = data_path + "\\" + "train.conll"


def generator_sentence():
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    cache = [['0', '<root>', '<root>', 'root', 'root', '_', '0', '核心成分']]
    for train_row in train_data.split("\n"):
        train_row = train_row.strip()
        if train_row == "":
            yield cache
            cache = [['0', '<root>', '<root>', 'root', 'root', '_', '0', '核心成分']]
        else:
            train_row_dep = train_row.split("\t")
            assert len(train_row_dep) == 8
            cache.append(train_row_dep)
    if len(cache) > 1:
        yield cache


sentence_list = []
for sentence in generator_sentence():
    sentence_row = [item[1] for item in sentence]
    sentence_list.append(sentence_row)

outp1 = "word2vec.model"
outp2 = "word2vec.json"
model = Word2Vec(sentences=sentence_list, size=100, window=5, min_count=1, workers=multiprocessing.cpu_count())
model.save(outp1)
# vector = model.wv
# import json
# with open(outp2, "w") as f:
#     f.write(json.dumps(vector, ensure_ascii=False))
# print(model.wv.vocab)
# print(model.wv["我"])
# for key in model.wv:
#     print(key)
