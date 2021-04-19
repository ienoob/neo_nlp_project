#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

from typing import List
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument
# from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, BertConfig

sample_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64
event_num = len(bd_data_loader.event2id)
batch_num = 10


def padding_data(input_batch):
    encodings = []
    event_labels = []
    event_trigger_start = []
    event_trigger_end = []
    max_len = 0
    for bd in input_batch:
        encodings.append(bd["encoding"])
        event_labels.append(bd["event_label"])
        max_len = max(max_len, len(bd["encoding"]))
    for bd in input_batch:
        trigger_start = np.zeros(max_len)
        trigger_end = np.zeros(max_len)
        for s, e, eid in bd["trigger_loc"]:
            trigger_start[s] = eid
            trigger_end[e] = eid
        event_trigger_start.append(trigger_start)
        event_trigger_end.append(trigger_end)

    return {
        "encoding": tf.keras.preprocessing.sequence.pad_sequences(encodings, padding="post"),
        "event_label": tf.cast(event_labels, dtype=tf.int64),
        "event_trigger_start": tf.cast(event_trigger_start, dtype=tf.float32),
        "event_trigger_end": tf.cast(event_trigger_end, dtype=tf.float32)
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
    trigger_loc = []
    event_arguments = []
    for e in event_list:
        arguments = []
        label_data[e.id] = 1
        trigger_loc.append((e.trigger_start, e.trigger_start+len(e.trigger)-1, e.id))
        for e_a in e.arguments:
            arguments.append((e_a.start, e_a.start+len(e_a.argument)-1, bd_data_loader.argument_role2id[e_a.role]))
        event_arguments.append(arguments)

    return {
        "encoding": tf.cast(text_id, dtype=tf.int64),
        "event_label": tf.cast(label_data, dtype=tf.int64),
        "trigger_loc": trigger_loc,
        "event_arguments": event_arguments
    }


class UNModel(tf.keras.models.Model):

    def __init__(self):
        super(UNModel, self).__init__()
        # self.bert = TFBertModel.from_pretrained(config)
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.event_classifier = tf.keras.layers.Dense(event_num, activation="sigmoid")
        self.tagger_start = tf.keras.layers.Dense(event_num, activation="softmax")
        self.tagger_end = tf.keras.layers.Dense(event_num, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.embed(inputs)
        seq = self.lstm(x)

        last_seq = seq[:,-1,:]
        last_seq = self.drop_out(last_seq)
        event_label = self.event_classifier(last_seq)
        tagger_start = self.tagger_start(seq)
        tagger_end = self.tagger_end(seq)

        return event_label, tagger_start, tagger_end

um_model = UNModel()
optimizer = tf.keras.optimizers.Adam()
# loss = tf.keras.losses.mean_squared_error()

sample_predict = tf.constant([[0.9, 0.5]])
sample_label = tf.constant(([[1, 0]]))

# print(tf.keras.losses.MSE(sample_label, sample_predict))
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_step(input_xx, input_yy, input_trigger_start, input_trigger_end):

    with tf.GradientTape() as tape:
        logits1, logits2, logits3  = um_model(input_xx)
        lossv1 = tf.reduce_mean(tf.keras.losses.MSE(input_yy, logits1))
        lossv2 = loss_func(input_trigger_start, logits2)
        lossv3 = loss_func(input_trigger_end, logits3)
        lossv = lossv1 + lossv2 + lossv3

    variables = um_model.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv


epoch = 100

for ep in range(epoch):

    for batch, data in enumerate(get_batch_data(batch_num)):
        loss_value = train_step(data["encoding"], data["event_label"],
                                data["event_trigger_start"],
                                data["event_trigger_end"])

        if batch % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss_value))
