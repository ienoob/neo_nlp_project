#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import metrix


msra_data = LoadMsraDataV2("D:\data\\ner\msra_ner_token_level\\")


char2id = {"pad": 0, "unk": 1}
max_len = -1
msra_train_id = []
for sentence in msra_data.train_sentence_list:
    sentence_id = []
    for s in sentence:
        if s not in char2id:
            char2id[s] = len(char2id)
        sentence_id.append(char2id[s])
    if len(sentence_id) > max_len:
        max_len = len(sentence_id)
    msra_train_id.append(torch.LongTensor(sentence_id))

tag_list = msra_data.train_tag_list
label2id = {"pad": 0, "O": 1}
for lb in msra_data.labels:
    if lb not in label2id:
        label2id[lb] = len(label2id)
id2label = {v:k for k, v in label2id.items()}
msra_tag_id = []
for tag in tag_list:
    tag_ids = []
    for tg in tag:
        tag_ids.append(label2id[tg])
    msra_tag_id.append(torch.LongTensor(tag_ids))

word_num = len(char2id)+1
embed_size = 64
rnn_dim = 64
class_num = len(label2id)
batch_size = 10


msra_train_id = pad_sequence(msra_train_id)
# msra_train_id = msra_train_id.transpose(0, 1)
msra_tag_id = pad_sequence(msra_tag_id)
# print(msra_train_id.size())
# train_data, target_batch = torch.Tensor(msra_train_id), torch.LongTensor(msra_tag_id)
dataset = Data.TensorDataset(msra_train_id, msra_tag_id)
loader = Data.DataLoader(dataset, batch_size, True)


class RNNNer(nn.Module):

    def __init__(self):
        super(RNNNer, self).__init__()

        self.embed = nn.Embedding(word_num, embed_size)
        self.lstm = nn.LSTM(embed_size, rnn_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.ner_classifier = nn.Linear(rnn_dim, class_num)

    def forward(self, input_x):
        mask = input_x.eq(0)
        input_x = input_x.transpose(0, 1)
        char_embed = self.embed(input_x)
        lstm_value, _ = self.lstm(char_embed)
        # lstm_value = self.drop(lstm_value)
        ner_logits = self.ner_classifier(lstm_value)

        return ner_logits

rnn_ner = RNNNer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn_ner.parameters(), lr=0.001)

Epoch = 100
for epoch in range(Epoch):
    for x, y in loader:
        pred = rnn_ner(x)

        loss = criterion(pred, y)

        break

        # if (epoch+1)%1000==0:
        #     print("Epoch:", "%04d" % (epoch + 1), "cost = ", "{:.6f}".format(loss))
        #
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    break
