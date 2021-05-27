#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/26 22:49
    @Author  : jack.li
    @Site    : 
    @File    : hmm_model.py

"""
import numpy as np
from nlp_applications.data_loader import LoadMsraDataV1, LoadMsraDataV2

msra_data = LoadMsraDataV2("D:\data\\nlp\\命名实体识别\\msra_ner_token_level\\")

label2id = msra_data.label2id
# label2id = {"O": 0}
# for lb in msra_data.labels:
#     if lb not in label2id:
#         label2id[lb] = len(label2id)


char2id = {"unk": 0}
char_count = {}
max_len = -1
msra_train_id = []
for sentence in msra_data.train_sentence_list:
    for s in sentence:
        if s not in char2id:
            char2id[s] = len(char2id)
        char_count.setdefault(s, 0)
        char_count[s] += 1
threshold = 20
cv = 0
char2id = {"unk": 0}
for k, v in char_count.items():
    if v < threshold:
        cv += 1
    else:
        char2id[k] = len(char2id)

print(cv)
n_label = len(label2id)
n_observer = len(char2id)
transferring_matrix = np.zeros((n_label, n_label))
trainsmit_matrix = np.zeros((n_observer, n_label))

init_transferring = np.zeros((n_label, 1))

for i, olist in enumerate(msra_data.train_sentence_list):
    for j, o in enumerate(olist):
        assert len(olist) == len(msra_data.train_tag_list[i])
        label_id = label2id[msra_data.train_tag_list[i][j]]
        o_id = char2id.get(o, 0)

        trainsmit_matrix[o_id][label_id] += 1

transferring_matrix_sum = transferring_matrix.sum(axis=1)
transferring_matrix_sum = transferring_matrix_sum.reshape((n_label, 1))

trainsmit_matrix_sum = trainsmit_matrix.sum(axis=1)
trainsmit_matrix_sum = trainsmit_matrix_sum.reshape((n_observer, 1))

transferring_matrix = transferring_matrix/transferring_matrix_sum
trainsmit_matrix = trainsmit_matrix/trainsmit_matrix_sum

init_transferring = init_transferring/init_transferring.sum()


