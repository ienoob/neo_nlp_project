#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/24 23:31
    @Author  : jack.li
    @Site    : 
    @File    : tplink.py

    TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking

"""
import tensorflow as tf


class Tplink(tf.keras.Model):

    def __init__(self, char_size, char_embed_size, word_size, word_char_size, lstm_size, relation_num):
        super(Tplink, self).__init__()
        self.embed = tf.keras.layers.Embedding(char_size, char_embed_size)
        self.word_embed = tf.keras.layers.Embedding(word_size, word_char_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))

        self.eh_eh_u = tf.keras.layers.Dense(char_embed_size)
        self.eh_eh_v = tf.keras.layers.Dense(char_embed_size)
        self.eh_eh_uv = tf.keras.layers.Dense(char_embed_size)
        self.entity_classifier = tf.keras.layers.Dense(1, activation="sigmoid")

        self.h_h_u = tf.keras.layers.Dense(char_embed_size)
        self.h_h_v = tf.keras.layers.Dense(char_embed_size)
        self.h_h_uv = tf.keras.layers.Dense(char_embed_size)
        self.h2h_relation = tf.keras.layers.Dense(relation_num, activation="softmax")

        self.t_t_u = tf.keras.layers.Dense(char_embed_size)
        self.t_t_v = tf.keras.layers.Dense(char_embed_size)
        self.t_t_uv = tf.keras.layers.Dense(char_embed_size)
        self.t2t_relation = tf.keras.layers.Dense(relation_num, activation="softmax")

    def call(self, inputs, input_words, input_entity_label, input_max_len, training=None, mask=None):
        mask_value = tf.math.logical_not(tf.math.equal(inputs, 0))
        char_embed = self.embed(inputs)
        word_embed = self.word_embed(input_words)

        embed = tf.concat([char_embed, word_embed], axis=-1)

        lstm_encoder = self.bi_lstm(embed, mask=mask_value)

        B, L, H = lstm_encoder.shape
        if L is None:
            L = input_max_len

        eh_u = tf.expand_dims(self.eh_eh_u(lstm_encoder), axis=1)
        eh_u = tf.keras.activations.relu(tf.tile(eh_u, multiples=(1, L, 1, 1)))
        eh_v = tf.expand_dims(self.eh_eh_v(lstm_encoder), axis=2)
        eh_v = tf.keras.activations.relu(tf.tile(eh_v, multiples=(1, 1, L, 1)))
        eh_uv = self.eh_eh_uv(tf.concat((eh_u, eh_v), axis=-1))

        eh_logits = self.entity_classifier(eh_uv)

        if not training:
            input_entity_value = tf.cast(tf.math.greater_equal(eh_logits, 0.5), dtype=tf.float32)
        else:
            input_entity_value = tf.cast(tf.expand_dims(input_entity_label, axis=-1), dtype=tf.float32)

        # relation_uv = tf.reduce_max(tf.reduce_max(eh_uv * input_entity_value, axis=1), axis=1)
        # relation_uv = tf.expand_dims(tf.expand_dims(relation_uv, 1))
        # relation_uv = tf.repeat(relation_uv, L, axis=1)
        # relation_uv = tf.repeat(relation_uv, L, axis=2)
        # print(relation_uv.shape)

        hh_u = tf.expand_dims(self.h_h_u(lstm_encoder), axis=1)
        hh_u = tf.keras.activations.relu(tf.tile(hh_u, multiples=(1, L, 1, 1)))
        hh_v = tf.expand_dims(self.h_h_v(lstm_encoder), axis=2)
        hh_v = tf.keras.activations.relu(tf.tile(hh_v, multiples=(1, 1, L, 1)))
        hh_uv = self.h_h_uv(tf.concat((hh_u, hh_v), axis=-1))
        # hh_uv = tf.concat((relation_uv, hh_uv), axis=-1)
        # hh_uv = hh_uv + relation_uv

        hh_logtis = self.h2h_relation(hh_uv)

        tt_u = tf.expand_dims(self.t_t_u(lstm_encoder), axis=1)
        tt_u = tf.keras.activations.relu(tf.tile(tt_u, multiples=(1, L, 1, 1)))
        tt_v = tf.expand_dims(self.t_t_v(lstm_encoder), axis=2)
        tt_v = tf.keras.activations.relu(tf.tile(tt_v, multiples=(1, 1, L, 1)))
        tt_uv = self.t_t_uv(tf.concat((tt_u, tt_v), axis=-1))
        # tt_uv = tf.concat((relation_uv, tt_uv), axis=-1)

        tt_logtis = self.t2t_relation(tt_uv)

        return eh_logits, hh_logtis, tt_logtis
