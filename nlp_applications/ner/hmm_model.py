#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/26 22:49
    @Author  : jack.li
    @Site    : 
    @File    : hmm_model.py

"""
import numpy as np
from nlp_applications.ner.evaluation import metrix_v2
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
msra_train_id = []
for sentence in msra_data.train_sentence_list:
    for s in sentence:
        if s not in char2id:
            char2id[s] = len(char2id)
        char_count.setdefault(s, 0)
        char_count[s] += 1

threshold = 0
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


# transferring_matrix = np.ones((n_label, n_label)) * 1e-10
# trainsmit_matrix = np.ones((n_label, n_observer)) * 1e-10
#
# init_transferring = np.ones((n_label, 1)) * 1e-10


class HMMModel(object):

    def __init__(self, observer_num, status_num, char2id, label2id):
        self.char2id = char2id
        self.label2id = label2id
        self.transferring_matrix = np.ones((status_num, status_num)) * 1e-10
        self.trainsmit_matrix = np.ones((status_num, observer_num)) * 1e-10

        self.init_transferring = np.ones((status_num, 1)) * 1e-10

    # def initialize(self, observer_list, status_list):
    #     char2id = {"unk": 0}
    #     label2id = {}
    #     char_count = {}
    #     for sentence in observer_list:
    #         for s in sentence:
    #             if s not in char2id:
    #                 char2id[s] = len(char2id)
    #             char_count.setdefault(s, 0)
    #             char_count[s] += 1
    #     for status in status_list:
    #         if s in status:
    #             if s not in label2id:
    #                 label2id[s] = len(label2id)
    #     n_label = len(label2id)
    #     n_observer = len(char2id)


    def fit(self, observer_list, status_list):
        for i, olist in enumerate(observer_list):
            for j, o in enumerate(olist):
                assert len(olist) == len(status_list[i])
                label_id = self.label2id[status_list[i][j]]
                o_id = self.char2id.get(o, 0)

                self.trainsmit_matrix[label_id][o_id] += 1

        for label_list in status_list:
            first_v_id = self.label2id[label_list[0]]
            self.init_transferring[first_v_id][0] += 1
            for j, labelv in enumerate(label_list[:-1]):
                labelv_id = self.label2id[labelv]
                next_labelv_id = self.label2id[label_list[j + 1]]
                self.transferring_matrix[labelv_id][next_labelv_id] += 1

        transferring_matrix_sum = self.transferring_matrix.sum(axis=1)
        transferring_matrix_sum = transferring_matrix_sum.reshape((n_label, 1))
        self.transferring_matrix = self.transferring_matrix / transferring_matrix_sum


        trainsmit_matrix_sum = self.trainsmit_matrix.sum(axis=1)
        trainsmit_matrix_sum = trainsmit_matrix_sum.reshape((n_label, 1))
        self.trainsmit_matrix = self.trainsmit_matrix / trainsmit_matrix_sum

        self.init_transferring = self.init_transferring / self.init_transferring.sum()

    def predict(self, sentence):
        sentence_id = [self.char2id.get(char, 0) for char in sentence]

        first = sentence_id[0]
        first_id = self.char2id.get(first, 0)
        state = self.trainsmit_matrix[:,first_id:first_id+1]
        first_state = state * self.init_transferring

        path = np.zeros((n_label, len(sentence)))
        path_score = np.zeros((n_label, len(sentence)))
        path_score[:,:1] = first_state
        j = 1
        for ob in sentence_id[1:]:
            ob_id = self.char2id.get(ob, 0)
            ob_state = self.trainsmit_matrix[:,ob_id:ob_id+1]
            last_score = path_score[:, j-1:j]

            ob_n_score = np.zeros((n_label, 1))
            for i in range(n_label):
                ob_score = last_score * self.transferring_matrix[:,i:i+1]
                ob_path_id = ob_score.argmax()
                path[i][j] = ob_path_id
                ob_n_score[i][0] = ob_score[ob_path_id][0]
            path_score[:, j:j+1] = ob_state * ob_n_score
            j += 1
        v = path_score[:,-1].argmax()
        # print(path_score[v][-1])
        final_path = [v]
        sentence_len = len(sentence)
        for j in range(sentence_len-1, 0, -1):
            v = int(path[v][j])
            final_path.append(v)
        assert len(sentence) == len(final_path)
        return [id2label[x] for x in final_path[::-1]]

    def predict_list(self, input_sentence_list):
        return [self.predict(sentence) for sentence in input_sentence_list]


model = HMMModel(n_observer, n_label, char2id, label2id)

model.fit(msra_data.train_sentence_list, msra_data.train_tag_list)
predict_value = model.predict_list(msra_data.test_sentence_list)
print(metrix_v2(msra_data.test_tag_list, predict_value))
