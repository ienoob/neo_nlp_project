#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import multiprocessing
from gensim.models import Word2Vec
from gensim import corpora

# print(model.wv.vocab)


class WVModel(object):

    def __init__(self):
        self.model = Word2Vec(size=256, window=5, min_count=1, workers=multiprocessing.cpu_count(), iter=10)

    def fit(self, data_list):
        self.model.build_vocab(data_list)
        self.model.train(data_list, total_examples=self.model.corpus_count, epochs=self.model.epochs)

