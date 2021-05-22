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
        self.ent_embed = tf.keras.layers.Embedding()

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10, return_sequences=True))

        self.emission = tf.keras.layers.Dense(entity_num)

        self.crf = None
        self.selection_u = tf.keras.layers.Dense(rel_num)
        self.selection_v = tf.keras.layers.Dense(rel_num)
        self.selection_uv = tf.keras.layers.Dense(rel_num)


    def call(self, char_ids, word_ids, entity_ids, training=None, mask=None):
        mask_value = tf.not_equal(char_ids, 0)
        char_embed = self.char_embed(char_ids)
        word_embed = self.word_embed(word_ids)

        embed = tf.concat([char_embed, word_embed], axis=-1)
        sent_encoder = self.bi_lstm(embed)
        eimission = self.emission(sent_encoder)

        ent_encoder = self.ent_emb(entity_ids)

        rel_encoder = tf.concat((sent_encoder, ent_encoder), axis=-1)
        B, L, H = rel_encoder.shape
        u = tf.expand_dims(self.selection_u(rel_encoder), axis=1)
        v = tf.expand_dims(self.selection_v(rel_encoder), axis=2)
        uv = self.selection_uv(tf.concat((u, v), axis=-1))

        selection_logits = tf.einsum('bijh,rh->birj', uv, self.rel_emb.weight)

def test_run_model():

    model = MultiHeaderModel()
    sample_char_id = tf.constant([[1, 2]])
    sample_word_id = tf.constant([[2, 3]])
    sample_entity_id = tf.constant([[1, 2]])










