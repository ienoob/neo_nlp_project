#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

from typing import List, Callable
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument
# from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, BertConfig

"""
    这个是一个baseline 模型，针对句子级别的数据
"""

sample_path = "D:\\data\\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64
event_num = len(bd_data_loader.event2id)
batch_num = 10
argument_num = len(bd_data_loader.argument_role2id)


def padding_data(input_batch):
    encodings = []
    event_labels = []
    event_trigger_start = []
    event_trigger_end = []
    trigger_masks = []
    event_argument_start = []
    event_argument_end = []
    event_label_ids = []
    max_len = 0
    for bd in input_batch:
        encodings.append(bd["encoding"])
        event_labels.append(bd["event_label"])
        max_len = max(max_len, bd["encoding"].shape[0])
        event_label_ids.append(bd["event_label_id"])

    for bd in input_batch:
        trigger_start = np.zeros(max_len)
        trigger_end = np.zeros(max_len)

        for s, e, eid in bd["trigger_loc"]:
            trigger_start[s] = eid
            trigger_end[e] = eid

            trigger_mask = np.ones(max_len)*-1e30
            trigger_mask[s:e+1] = 0
            trigger_masks.append(trigger_mask)

        event_trigger_start.append(trigger_start)
        event_trigger_end.append(trigger_end)

        for argus in bd["event_arguments"]:
            event_argus_start = np.zeros(max_len)
            event_argus_end = np.zeros(max_len)
            for s, e, aid in argus:
                event_argus_start[s] = aid
                event_argus_end[s] = aid
            event_argument_start.append(event_argus_start)
            event_argument_end.append(event_argus_end)

    return {
        "encoding": tf.keras.preprocessing.sequence.pad_sequences(encodings, padding="post"),
        "event_label": tf.cast(event_labels, dtype=tf.int64),
        "event_trigger_start": tf.cast(event_trigger_start, dtype=tf.float32),
        "event_trigger_end": tf.cast(event_trigger_end, dtype=tf.float32),
        "event_argument_start": tf.cast(event_argument_start, dtype=tf.float32),
        "event_argument_end": tf.cast(event_argument_end, dtype=tf.float32),
        "trigger_masks": tf.cast(event_argument_end, dtype=tf.float32),
        "event_label_ids": event_label_ids
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
    event_label_id = []
    for e in event_list:
        arguments = []
        label_data[e.id] = 1
        trigger_loc.append((e.trigger_start, e.trigger_start+len(e.trigger)-1, e.id))
        for e_a in e.arguments:
            arguments.append((e_a.start, e_a.start+len(e_a.argument)-1, bd_data_loader.argument_role2id[e_a.role]))
        event_arguments.append(arguments)
        event_label_id.append(e.id)

    return {
        "encoding": tf.cast(text_id, dtype=tf.int64),
        "event_label": tf.cast(label_data, dtype=tf.int64),
        "trigger_loc": trigger_loc,
        "event_arguments": event_arguments,
        "event_label_id": event_label_id
    }


# class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
#
#     def __init__(
#         self,
#         initial_learning_rate: float,
#         decay_schedule_fn: Callable,
#         warmup_steps: int,
#         power: float = 1.0,
#         name: str = None,
#     ):
#         super().__init__()
#         self.initial_learning_rate = initial_learning_rate
#         self.warmup_steps = warmup_steps
#         self.power = power
#         self.decay_schedule_fn = decay_schedule_fn
#         self.name = name
#
#     def __call__(self, step):
#         with tf.name_scope(self.name or "WarmUp") as name:
#             # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
#             # learning rate will be `global_step/num_warmup_steps * init_lr`.
#             global_step_float = tf.cast(step, tf.float32)
#             warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
#             warmup_percent_done = global_step_float / warmup_steps_float
#             warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
#             return tf.cond(
#                 global_step_float < warmup_steps_float,
#                 lambda: warmup_learning_rate,
#                 lambda: self.decay_schedule_fn(step - self.warmup_steps),
#                 name=name,
#             )
#
#     def get_config(self):
#         return {
#             "initial_learning_rate": self.initial_learning_rate,
#             "decay_schedule_fn": self.decay_schedule_fn,
#             "warmup_steps": self.warmup_steps,
#             "power": self.power,
#             "name": self.name,
#         }

boundaries = [100000, 110000]
values = [0.001, 0.0001, 0.00001]

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

def repeat_data(input_feature, input_event_label_ids):
    inner_batch_num = input_feature.shape[0]
    out_data = []
    for i in range(inner_batch_num):
        sub_feature = input_feature[i]
        for j in input_event_label_ids[i]:
            out_data.append(sub_feature)

    return tf.cast(out_data, dtype=tf.float32)


def get_data_loc(input_trigger_start, input_trigger_end, input_seq):

    input_trigger_start_ind = tf.argmax(input_trigger_start, axis=2)
    input_trigger_end_ind = tf.argmax(input_trigger_end, axis=2)
    batch_num = input_trigger_start_ind.shape[0]
    batch_len = input_trigger_start_ind.shape[1]

    span_max = 10
    event_res = []
    arg_feature = []
    arg_mask = []
    for i in range(batch_num):
        start_one = input_trigger_start_ind[i]
        end_one = input_trigger_end_ind[i]
        for j, x in enumerate(start_one):
            if x == 0:
                continue
            for k, y in enumerate(end_one):
                if k < j:
                    continue
                if k-j > span_max:
                    continue
                if x == y:
                    event_res.append((i, j, k, x.numpy()))
                    arg_feature.append(input_seq[i])
                    event_one_mask = np.ones(batch_len)*-1e30
                    event_one_mask[j:k+1] = 0
                    arg_mask.append(event_one_mask)
    arg_feature = tf.cast(arg_feature, dtype=tf.float32)
    arg_mask = tf.cast(arg_mask, dtype=tf.float32)
    e_m = tf.expand_dims(arg_mask, axis=-1)
    mask_arg = arg_feature+e_m

    argument_feature = tf.concat([arg_feature, mask_arg], axis=2)
    return event_res, argument_feature


def get_start_end(input_start, input_end):
    input_trigger_start_ind = tf.argmax(input_start, axis=2)
    input_trigger_end_ind = tf.argmax(input_end, axis=2)
    batch_num = input_trigger_start_ind.shape[0]

    span_max = 10
    res = []
    for i in range(batch_num):
        start_one = input_trigger_start_ind[i]
        end_one = input_trigger_end_ind[i]
        for j, x in enumerate(start_one):
            if x == 0:
                continue
            for k, y in enumerate(end_one):
                if k < j:
                    continue
                if k-j > span_max:
                    continue
                if x == y:
                    res.append((i, j, k, x.numpy()))
    return res


class UNModel(tf.keras.models.Model):

    def __init__(self):
        super(UNModel, self).__init__()
        # self.bert = TFBertModel.from_pretrained(config)
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.event_classifier = tf.keras.layers.Dense(event_num, activation="sigmoid")
        self.tagger_start = tf.keras.layers.Dense(event_num, activation="softmax")
        self.tagger_end = tf.keras.layers.Dense(event_num, activation="softmax")

        self.argument_start = tf.keras.layers.Dense(argument_num, activation="softmax")
        self.argument_end = tf.keras.layers.Dense(argument_num, activation="softmax")

    def call(self, inputs, trigger_masks=None, event_label_ids=None, training=None, mask=None):
        x = self.embed(inputs)
        seq = self.lstm(x)

        last_seq = seq[:,-1,:]
        last_seq = self.drop_out(last_seq)
        event_label = self.event_classifier(last_seq)
        tagger_start = self.tagger_start(seq)
        tagger_end = self.tagger_end(seq)

        m = tf.expand_dims(trigger_masks, axis=-1)
        argument_embed = repeat_data(seq, event_label_ids)
        mask_argument = argument_embed+m

        argument_feature = tf.concat([argument_embed, mask_argument], axis=2)
        argument_start = self.argument_start(argument_feature)
        argument_end = self.argument_end(argument_feature)
        return event_label, tagger_start, tagger_end, argument_start, argument_end

    def predict(self, input_encoding):
        x = self.embed(input_encoding)
        seq = self.lstm(x)

        last_seq = seq[:, -1, :]
        last_seq = self.drop_out(last_seq, training=True)
        event_label = self.event_classifier(last_seq)

        trigger_start = self.tagger_start(seq)
        trigger_end = self.tagger_end(seq)
        event_res, argument_feature = get_data_loc(trigger_start, trigger_end, seq)

        argument_start = self.argument_start(argument_feature)
        argument_end = self.argument_end(argument_feature)

        arg_extract = get_start_end(argument_start, argument_end)

        event_info = dict()
        for i, x in enumerate(event_res):
            x_id = x[0]
            event_info.setdefault(x_id, dict())





um_model = UNModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# loss = tf.keras.losses.mean_squared_error()

sample_predict = tf.constant([[0.9, 0.5]])
sample_label = tf.constant(([[1, 0]]))

# print(tf.keras.losses.MSE(sample_label, sample_predict))
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_step(input_xx, input_yy, input_trigger_mask, input_event_label_ids, input_trigger_start, input_trigger_end,
               input_argument_start, input_argument_end):

    with tf.GradientTape() as tape:
        logits1, logits2, logits3, logits4, logits5 = um_model(input_xx, input_trigger_mask, input_event_label_ids)
        lossv1 = tf.reduce_mean(tf.keras.losses.MSE(input_yy, logits1))
        lossv2 = loss_func(input_trigger_start, logits2)
        lossv3 = loss_func(input_trigger_end, logits3)
        lossv4 = loss_func(input_argument_start, logits4)
        lossv5 = loss_func(input_argument_end, logits5)
        lossv = lossv1 + lossv2 + lossv3 + lossv4 + lossv5

    variables = um_model.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv


epoch = 100

for ep in range(epoch):

    for batch, data in enumerate(get_batch_data(batch_num)):
        loss_value = train_step(data["encoding"], data["event_label"],
                                data["trigger_masks"],
                                data["event_label_ids"],
                                data["event_trigger_start"],
                                data["event_trigger_end"],
                                data["event_argument_start"],
                                data["event_argument_end"]
                                )
        um_model.predict(data["encoding"])

        if batch % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss_value))
        break
    break
