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

    def __init__(self, char_size, char_embed_size, lstm_size):
        super(Tplink, self).__init__()
        self.embed = tf.keras.layers.Embedding(char_size, char_embed_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))

        self.eh_eh_u = tf.keras.layers.Dense(char_embed_size)
        self.eh_eh_v = tf.keras.layers.Dense(char_embed_size)
        self.eh_eh_uv = tf.keras.layers.Dense(char_embed_size)

        self.entity_classifier = tf.keras.layers.Dense(1, activation="sigmoid")



    def call(self, inputs, training=None, mask=None):
        mask_value = tf.math.logical_not(tf.math.equal(inputs, 0))
        char_embed = self.char_embed(inputs)

        lstm_encoder = self.bi_lstm(char_embed, mask=mask_value)

        B, L, H = lstm_encoder.shape
        eh_u = tf.expand_dims(self.eh_eh_u(lstm_encoder), axis=1)
        eh_u = tf.keras.activations.relu(tf.tile(eh_u, multiples=(1, L, 1, 1)))
        eh_v = tf.expand_dims(self.eh_eh_v(lstm_encoder), axis=2)
        eh_v = tf.keras.activations.relu(tf.tile(eh_v, multiples=(1, 1, L, 1)))
        eh_uv = self.eh_eh_uv(tf.concat((eh_u, eh_v), axis=-1))

        eh_logits = self.entity_classifier(eh_uv)







