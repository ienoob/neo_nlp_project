#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument

"""
    失败的树状输出
"""


sample_path = "D:\\data\\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64
event_num = len(bd_data_loader.event2id)
batch_num = 10
argument_num = len(bd_data_loader.argument_role2id)



def seq_and_vec(seq, vec):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    vec = tf.expand_dims(vec, 1)
    vec = tf.zeros_like(seq[:, :, :1]) + vec
    return tf.concat([seq, vec], 2)


class EventModelV2(tf.keras.Model):

    def __init__(self, char_size, char_embed, lstm_size, event_class, event_embed_size, event2argument):
        super(EventModelV2, self).__init__()
        self.embed = tf.keras.layers.Embedding(char_size, char_embed)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.event_embed = tf.keras.layers.Embedding(event_class, event_embed_size)
        self.event_type = tf.keras.layers.Dense(event_class, activation="sigmoid")
        self.event_arguments = [tf.keras.layers.Dense(len(event2argument[i])) for i in range(event_class) if i]

    def call(self, inputs, input_event_type, training=None, mask=None):
        mask = tf.logical_not(tf.equal(inputs, 0))
        embed = self.embed(inputs)
        bi_lstm_value = self.bi_lstm(embed, mask=mask)
        event_logits = self.event_type(bi_lstm_value[:, -1, :])
        event_embed_value = self.event_embed(input_event_type)
        event_feature = seq_and_vec(bi_lstm_value, event_embed_value[:, 0, :])

        event_arguments_logits = []
        batch_num_v = inputs.shape[0]
        # print(input_event_type)
        for b in range(batch_num_v):
            print(input_event_type[b])
            event_arguments_logits.append(self.event_arguments[input_event_type[b][0]](event_feature))
        event_arguments_logits = tf.cast(event_arguments_logits, dtype=tf.float32)

        return event_logits, event_arguments_logits, mask







