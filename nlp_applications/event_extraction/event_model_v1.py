#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

from typing import List
import numpy as np
import tensorflow as tf
from baidu2021duee.data_loader import Baidu2021dueeDataloader, EventDocument, Event
from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, BertConfig

sample_path = "D:\data\句子级事件抽取\\"
bd_data_loader = Baidu2021dueeDataloader(sample_path)

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64
event_num = len(bd_data_loader.event2id)
batch_num = 10


def padding_data(input_batch):
    encodings = []
    event_labels = []
    event_triggers = []
    for bd in input_batch:
        encodings.append(bd["encoding"])
        event_labels.append(bd["event_label"])

    return {
        "encoding": tf.keras.preprocessing.sequence.pad_sequences(encodings, padding="post"),
        "event_label": tf.cast(event_labels, dtype=tf.int64)
    }


def get_batch_data(input_batch_num: int):
    batch_list = []
    for doc in bd_data_loader.document:
        deal_data = sample_single_doc(doc)
        batch_list.append(deal_data)
        if len(batch_list) == input_batch_num:
            yield padding_data(batch_list)
            batch_list = []


def sample_single_doc(input_doc: EventDocument):
    text_id = input_doc.text_id
    event_list: List[Event] = input_doc.event_list
    label_data = np.zeros(event_num)
    for e in event_list:
        label_data[e.id] = 1

    return {
        "encoding": tf.cast(text_id, dtype=tf.int64),
        "event_label": tf.cast(label_data, dtype=tf.int64)
    }


class UNModel(tf.keras.models.Model):

    def __init__(self):
        super(UNModel, self).__init__()
        # self.bert = TFBertModel.from_pretrained(config)
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
        self.event_classifier = tf.keras.layers.Dense(event_num, activation="sigmoid")
        self.tagger_indx = tf.keras.layers.Dense(event_num*2, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        x = self.embed(inputs)
        seq = self.lstm(x)

        last_seq = seq[:,-1,:]
        event_label = self.event_classifier(last_seq)
        # tagger_indx = self.tagger_indx(seq)

        return event_label


um_model = UNModel()
optimizer = tf.keras.optimizers.Adam()
# loss = tf.keras.losses.mean_squared_error()

sample_predict = tf.constant([[0.9, 0.5]])
sample_label = tf.constant(([[1, 0]]))

print(tf.keras.losses.MSE(sample_label, sample_predict))


def train_step(input_xx, input_yy):

    with tf.GradientTape() as tape:
        logits = um_model(input_xx)
        lossv = tf.reduce_mean(tf.keras.losses.MSE(input_yy, logits))

    variables = um_model.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv


epoch = 100

for ep in range(epoch):

    for batch, data in enumerate(get_batch_data(batch_num)):
        loss_value = train_step(data["encoding"], data["event_label"])

        if batch % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss_value))
