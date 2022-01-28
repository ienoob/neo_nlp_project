#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import time
import jieba
import numpy as np


data_path = "D:\\data\\语料库\\sohu-20091019-20130819\\sohu-20091019-20130819\\20091019"


class SimpleEngine(object):

    def __init__(self, input_path):
        self.data_path = input_path
        self.word2doc = dict()
        self.word_df = dict()

    def bm25_score(self):
        pass

    def build_engine(self):
        with open(self.data_path, "r", encoding="utf-8", errors='ignore') as f:
            data = f.read()

        data_list = data.split("\n")
        stop_words = set()

        word2doc = dict()
        words_list = []
        match_list = ["news", "sports", "business", "yule", "cul", "auto", "mil", "mil.news", "music.yule"]
        for i, item in enumerate(data_list):
            content = ""
            iv = 0
            while True:
                state = 0

                m = re.finditer("http://"+match_list[iv]+".sohu.com/[\d]{8}/n[\d]{9}.shtml", item)
                for k in m:
                    content = item[k.end():]
                    state = 1
                if state or iv == len(match_list)-1:
                    break
                iv += 1
            if content == "":
                print(item)
                break

            word_tf_dict = dict()
            n_len = 0
            for word in jieba.cut(content):
                if word in stop_words:
                    continue
                word_tf_dict.setdefault(word, 0)
                word_tf_dict[word] += 1
                n_len += 1

            for word, tf in word_tf_dict.items():
                self.word2doc.setdefault(word, [])
                self.word2doc[word].append((tf*1.0/n_len, i))

        for k, v in self.word2doc.items():
            self.word2doc[k] = sorted(v, reverse=True)

    def search(self, word, limit=10):
        return self.word2doc.get(word, [])[:limit]


engine = SimpleEngine(data_path)
engine.build_engine()
start = time.time()
print(engine.search("亚洲"))
print("cost {}".format(time.time()-start))
