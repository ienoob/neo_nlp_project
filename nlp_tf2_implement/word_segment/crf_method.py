#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from nlp_applications.ner.crf_model import CRFNerModel
from nlp_applications.word_segment.load_words import generator_seg_sentence
from nlp_applications.ner.evaluation import metrix_v2


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

data_path = "D:\data\depency_parser\evsam05\依存分析训练数据\THU"
train_data_path = data_path + "\\" + "train.conll"
dev_data_path = data_path + "\\" + "dev.conll"

data_res, data_len = generator_seg_sentence(train_data_path)
dev_data_res, _ = generator_seg_sentence(dev_data_path)


def f(input_data):
    label = []
    for x in input_data["seg_data"]:
        label.append("B-B")
        label += ["I-B" for _ in x[1:]]
    return label

train_data = [data["raw_data"] for data in data_res]
train_data_label = [f(data) for data in data_res]

test_data = [data["raw_data"] for data in data_res]
test_data_label = [f(data) for data in data_res]

X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data_label]

X_test = [sent2features(s) for s in test_data]
y_test = [sent2labels(s) for s in test_data_label]

# print(X_train)
print(len(y_train))

crf_mode = CRFNerModel()
# crf_mode.load_model()
crf_mode.fit(X_train, y_train)

predict_labels = crf_mode.predict_list(X_test)

print(metrix_v2(y_test, predict_labels))
