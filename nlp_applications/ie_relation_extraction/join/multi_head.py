#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/11 19:36
    @Author  : jack.li
    @Site    : 
    @File    : multi_head.py

    Joint entity recognition and relation extraction as a multi-head selection problem

"""
import tensorflow as tf


char_size = 10
char_embed = 10
word_embed = 10
entity_num = 10
rel_num = 10


class MultiHeaderModel(tf.keras.Model):

    def __init__(self):
        super(MultiHeaderModel, self).__init__()

        self.char_embed = tf.keras.layers.Embedding(char_size, char_embed)
        self.word_embed = tf.keras.layers.Embedding(char_size, char_embed)

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10, return_sequences=True))

        self.emission = tf.keras.layers.Dense(entity_num)

        self.crf = None
        self.selection_u = tf.keras.layers.Dense(rel_num)
        self.selection_v = tf.keras.layers.Dense(rel_num)
        self.selection_uv = tf.keras.layers.Dense(rel_num)



    def call(self, char_ids, word_ids, training=None, mask=None):
        mask_value = tf.not_equal(char_ids, 0)
        char_embed = self.char_embed(char_ids)
        word_embed = self.word_embed(word_ids)

        embed = tf.concat([char_embed, word_embed], axis=-1)
        sent_encoder = self.bi_lstm(embed)






