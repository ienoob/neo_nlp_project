#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np

def load_word_vector(embedding_file):
    embedding_dict = dict()
    with open(embedding_file, encoding="utf-8") as f:
        for line in f:
            if len(line.rstrip().split(" ")) <= 2:
                continue
            token, vector = line.rstrip().split(" ", 1)
            embedding_dict[token] = np.fromstring(vector, dtype=np.float, sep=" ")
    return embedding_dict


# word_embed_path = "D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char"
# embed_dict = load_word_vector(word_embed_path)
# for k, v in embed_dict.items():
#     print(k, v)
#     break

# print(np.random.randint(1, 1))
