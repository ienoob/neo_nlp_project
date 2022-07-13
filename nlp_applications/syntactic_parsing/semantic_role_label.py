#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/7/3 10:17
    @Author  : jack.li
    @Site    : 
    @File    : semantic_role_label.py

"""
from nlp_applications.data_loader import LoaderCPBSRL
from nlp_applications.ner.crf_model import CRFNerModel

data_path = "D:\\BaiduNetdiskDownload\\data"
data_loader = LoaderCPBSRL(data_path)

print("data loader prepared !")


def extract_bios(input_list):
    res = []
    start = -1
    state = 0
    for i, x in enumerate(input_list):
        sign = x[0]
        # print(sign, state, i)
        if sign == "S":
            res.append((i, i+1, x[2:]))
            start = i+1

            state = 0
        elif sign == "B":
            start = i
            state = 1
        elif sign == "I":
            if input_list[i-1][0] not in ["B", "I"]:
                state = 0
            if input_list[i-1][2:] != x[2:]:
                state = 0
        elif sign == "E":
            if state:
                res.append((start, i+1, x[2:]))
                start = i+1
                state = 0
        elif sign == "O":
            state = 0
        else:
            res.append((i, i + 1, x))
            start = i + 1
            state = 0
    return res

# print(data_loader.labels[3])
# print(extract_bios(data_loader.labels[3]))

label_set = {l for la in data_loader.labels for l in la}
print(label_set)
def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word_lower()': word.lower(),
        "word_isdigit()": word.isdigit()
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isdigit()': word1.isdigit()
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.isdigit()': word1.isdigit()
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for label in sent]

X_train = [sent2features(s) for s in data_loader.documents]
y_train = [sent2labels(s) for s in data_loader.labels]

X_test = [sent2features(s) for s in data_loader.dev_documents]
y_test = [sent2labels(s) for s in data_loader.dev_labels]

crf_mode = CRFNerModel(verbose=True, max_iterations=200)
# crf_mode.load_model()
crf_mode.fit(X_train, y_train)
predict_labels = crf_mode.predict_list(X_test)
#
hit_num = 0.0
pre_num = 0.0
true_num = 0.0
for i, dev_l in enumerate(data_loader.dev_labels):

    real = extract_bios(dev_l)
    pred = extract_bios(predict_labels[i])

    hit_num += len(set(real) & set(pred))
    pre_num += len(pred)
    true_num += len(real)

print(hit_num, pre_num, true_num)

# predict_labels = crf_mode.predict_list(X_test)
