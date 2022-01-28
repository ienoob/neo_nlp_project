#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/31 22:10
    @Author  : jack.li
    @Site    : 
    @File    : novel_tagging.py

"""
import tensorflow as tf


class NovelTaggingModelPointerNet(tf.keras.Model):

    def __init__(self, char_num, char_embed, lstm_embed, relation_num):
        super(NovelTaggingModelPointerNet, self).__init__()

        self.char_embed = tf.keras.layers.Embedding(char_num, char_embed)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_embed, return_sequences=True))
        self.pointer_net = tf.keras.layers.Dense(relation_num*4, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        mask_value = tf.math.logical_not(tf.math.equal(inputs, 0))
        char_embed = self.char_embed(inputs)
        bi_value = self.bi_lstm(char_embed, mask=mask_value)
        logits = self.pointer_net(bi_value)

        return logits









