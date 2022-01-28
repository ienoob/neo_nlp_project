#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument

"""
    失败的树状输出,但是可以利用mask 来实现，不同event 和 argument 对应
"""


def seq_and_vec(seq, vec):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    vec = tf.expand_dims(vec, 1)
    vec = tf.zeros_like(seq[:, :, :1]) + vec
    return tf.concat([seq, vec], 2)


class EventModelV2(tf.keras.Model):

    def __init__(self, char_size, char_embed, lstm_size, event_class, event_embed_size, event_argument_num, event2argument_mask):
        super(EventModelV2, self).__init__()
        self.embed = tf.keras.layers.Embedding(char_size, char_embed)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.event_embed = tf.keras.layers.Embedding(event_class, event_embed_size)
        self.event_type = tf.keras.layers.Dense(event_class, activation="sigmoid")
        self.event_arguments = tf.keras.layers.Dense(event_argument_num)
        self.event2argument_mask = event2argument_mask

    def call(self, inputs, input_event_type=None, input_evnet_type_mask=None, training=None, mask=None):
        mask = tf.logical_not(tf.equal(inputs, 0))
        text_len = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-1)-1
        text_len = tf.expand_dims(text_len, axis=-1)
        embed = self.embed(inputs)
        bi_lstm_value = self.bi_lstm(embed, mask=mask)
        last_value = tf.gather_nd(bi_lstm_value, text_len, batch_dims=1)
        event_logits = self.event_type(last_value)
        if training:
            event_embed_value = self.event_embed(input_event_type)
            event_feature = seq_and_vec(bi_lstm_value, event_embed_value[:, 0, :])

            event_arguments_logits = self.event_arguments(event_feature)
            event_arguments_logits += input_evnet_type_mask
            event_arguments_logits = tf.keras.activations.softmax(event_arguments_logits, axis=-1)

            return event_logits, event_arguments_logits, mask
        else:
            batch_res = []
            for b, row_value in enumerate(event_logits):
                row_res = []
                for e_id, e_value in enumerate(row_value):
                    if e_value < 0.5:
                        continue
                    if e_id == 0:
                        continue
                    e_id_input = tf.cast([[e_id]], dtype=tf.int32)
                    event_embed_value = self.event_embed(e_id_input)
                    event_feature = seq_and_vec(bi_lstm_value[b:b+1,:,:], event_embed_value[:, 0, :])
                    event_arguments_logits = self.event_arguments(event_feature)
                    input_evnet_type_mask = self.event2argument_mask[e_id]
                    event_arguments_logits += tf.cast([input_evnet_type_mask], dtype=tf.float32)
                    event_arguments_logits = tf.keras.activations.softmax(event_arguments_logits, axis=-1)
                    event_arguments_argmax = tf.argmax(event_arguments_logits, axis=-1)[0]
                    row_res.append((e_id, event_arguments_argmax.numpy()))
                batch_res.append(row_res)
            return batch_res









