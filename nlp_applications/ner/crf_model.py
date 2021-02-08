#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/29 22:47
    @Author  : jack.li
    @Site    : 
    @File    : crf_model.py

"""
import pickle
from sklearn_crfsuite import CRF
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from nlp_applications.data_loader import LoadMsraDataV1, LoadMsraDataV2


def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word_lower()': word.lower()
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower()
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower()
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for label in sent]


# msra_data = LoadMsraDataV1("D:\data\\nlp\命名实体识别\data\msra")

msra_data = LoadMsraDataV2("D:\data\\nlp\\命名实体识别\\msra_ner_token_level\\")

print(msra_data.train_tag_list[0])

X_train = [sent2features(s) for s in msra_data.train_sentence_list]
y_train = [sent2labels(s) for s in msra_data.train_tag_list]

X_test = [sent2features(s) for s in msra_data.test_sentence_list]
y_test = [sent2labels(s) for s in msra_data.test_tag_list]

# print(X_train)
print(len(y_train))


class CRFNerModel(object):

    def __init__(self):
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.save_model = "crf.model"

    def fit(self, train_x, train_y):
        self.crf.fit(train_x, train_y)

        model_data = pickle.dumps(self.crf)
        with open(self.save_model, "wb") as f:
            f.write(model_data)

    def predict(self, input_x):
        input_x = list(input_x)
        input_feature = [sent2features(input_x)]
        return self.crf.predict(input_feature)

    def extract_ner(self, input_x):
        extract_ner = []
        res = self.predict(input_x)
        print(res)

        start = None
        label = None
        for i, x in enumerate(res[0]):
            if x == "O":
                if start is not None:
                    extract_ner.append((start, i, label, input_x[start:i]))
                    start = None
                    label = None
            else:
                xindex, xlabel = x.split("-")
                if xindex == "B":
                    if start is not None:
                        extract_ner.append((start, i, label, input_x[start:i]))
                    start = i
                    label = xlabel
                else:
                    if label != xlabel:
                        start = None
                        label = None
        return extract_ner


crf_mode = CRFNerModel()
crf_mode.fit(X_train, y_train)

print(crf_mode.extract_ner("1月18日，在印度东北部一座村庄，一头小象和家人走过伐木工人正在清理的区域时被一根圆木难住了。"))