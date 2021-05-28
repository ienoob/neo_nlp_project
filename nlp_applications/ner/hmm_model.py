#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/26 22:49
    @Author  : jack.li
    @Site    : 
    @File    : hmm_model.py

"""
import numpy as np
from nlp_applications.ner.evaluation import metrix
from nlp_applications.data_loader import LoadMsraDataV1, LoadMsraDataV2

msra_data = LoadMsraDataV2("D:\data\\nlp\\命名实体识别\\msra_ner_token_level\\")

label2id = msra_data.label2id
id2label = {v:k for k,v in label2id.items()}
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
threshold = 15
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
print(n_label, n_observer)


transferring_matrix = np.zeros((n_label, n_label))
trainsmit_matrix = np.zeros((n_label, n_observer))

init_transferring = np.zeros((n_label, 1))

for i, olist in enumerate(msra_data.train_sentence_list):
    for j, o in enumerate(olist):
        assert len(olist) == len(msra_data.train_tag_list[i])
        label_id = label2id[msra_data.train_tag_list[i][j]]
        o_id = char2id.get(o, 0)

        trainsmit_matrix[label_id][o_id] += 1

for label_list in msra_data.train_tag_list:
    first_v_id = label2id[label_list[0]]
    init_transferring[first_v_id][0] += 1
    for j, labelv in enumerate(label_list[:-1]):
        labelv_id = label2id[labelv]
        next_labelv_id = label2id[label_list[j+1]]
        transferring_matrix[labelv_id][next_labelv_id] += 1


transferring_matrix_sum = transferring_matrix.sum(axis=1)
transferring_matrix_sum = transferring_matrix_sum.reshape((n_label, 1))
transferring_matrix = transferring_matrix/transferring_matrix_sum

d = np.array([[1, 2, 5], [2, 3, 6]])
print(d.sum(axis=1))

trainsmit_matrix_sum = trainsmit_matrix.sum(axis=1)
trainsmit_matrix_sum = trainsmit_matrix_sum.reshape((n_label, 1))
trainsmit_matrix = trainsmit_matrix/trainsmit_matrix_sum

init_transferring = init_transferring/init_transferring.sum()



def predict(sentence):
    sentence_id = [char2id.get(char, 0) for char in sentence]

    first = sentence_id[0]
    first_id = char2id.get(first, 0)
    state = trainsmit_matrix[:,first_id:first_id+1]
    first_state = state * init_transferring

    path = np.zeros((n_label, len(sentence)))
    path_score = np.zeros((n_label, len(sentence)))
    path_score[:,:1] = first_state
    j = 1
    for ob in sentence_id[1:]:
        ob_id = char2id.get(ob, 0)
        ob_state = trainsmit_matrix[:,ob_id:ob_id+1]
        last_score = path_score[:, j-1:j]

        ob_n_score = np.zeros((n_label, 1))
        for i in range(n_label):
            ob_score = last_score * transferring_matrix[:,i:i+1]
            ob_path_id = ob_score.argmax()
            path[i][j] = ob_path_id
            ob_n_score[i][0] = ob_score[ob_path_id][0]
        path_score[:, j:j+1] = ob_state * ob_n_score
        j += 1
    v = path_score[:,-1].argmax()
    final_path = [v]
    sentence_len = len(sentence)
    for j in range(sentence_len-1, 0, -1):
        v = int(path[v][j])
        final_path.append(v)
    assert len(sentence) == len(final_path)
    return [id2label[x] for x in final_path[::-1]]

def predict_list(input_sentence_list):
    return [predict(sentence) for sentence in input_sentence_list]

predict_value = predict_list(msra_data.test_sentence_list)
print(metrix(msra_data.test_tag_list, predict_value))
