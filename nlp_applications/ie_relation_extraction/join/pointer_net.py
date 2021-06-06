#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/11 19:43
    @Author  : jack.li
    @Site    : 
    @File    : pointer_net.py

    Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy

"""
import jieba
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops



class ConditionalLayerNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = tf.ones(hidden_size)
        self.bias = tf.zeros(hidden_size)
        self.variance_epsilon = eps

        self.beta_dense = tf.keras.layers.Dense(hidden_size, bias=False)
        self.gamma_dense = tf.keras.layers.Dense(hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / tf.sqrt(s + self.variance_epsilon)
        return weight * x + bias


class PointerNet(tf.keras.models.Model):

    def __init__(self, vocab_size, embed_size, word_size, word_embed_size, lstm_size, predicate_num):
        super(PointerNet, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.word_embed = tf.keras.layers.Embedding(word_size, word_embed_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.sub_classifier = tf.keras.layers.Dense(2, activation="sigmoid")
        self.po_classifier = tf.keras.layers.Dense(predicate_num*2, activation="sigmoid")

    def call(self, inputs, word_ids, input_sub_span=None, training=None, mask=None):
        char_embed = self.embed(inputs)
        word_embed = self.word_embed(word_ids)

        embed = tf.concat([char_embed, word_embed], axis=-1)
        mask_value = math_ops.not_equal(inputs, 0)
        input_lstm_value = self.bi_lstm(embed, mask=mask_value)

        sub_preds = self.sub_classifier(input_lstm_value)
        if not training:
            input_sub_span = tf.where(tf.greater(sub_preds, 0.5), 1.0, 0.0)
            input_sub_span = input_sub_span[:,:,0] + input_sub_span[:,:,1]
            input_sub_span = tf.where(tf.greater(input_sub_span, 0.0), 1.0, 0.0)

        input_sub_span = tf.expand_dims(input_sub_span, axis=-1)

        input_sub_feature = tf.multiply(input_lstm_value, input_sub_span)
        input_po_feature = tf.concat([input_lstm_value, input_sub_feature], axis=-1)

        po_preds = self.po_classifier(input_po_feature)

        sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
        po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

        return sub_preds, po_preds, mask_value

    # def predict(self, inputs):
        # input_embed = self.embed(inputs)
        # mask_value = math_ops.not_equal(inputs, 0)
        # input_lstm_value = self.bi_lstm(input_embed, mask=mask_value)
        #
        # sub_preds = self.sub_classifier(input_lstm_value)
        #
        # input_sub_span = tf.where(tf.greater(sub_preds, 0.5), 1.0, 0.0)
        # input_sub_span = tf.expand_dims(input_sub_span, axis=-1)
        # input_sub_feature = tf.multiply(input_lstm_value, input_sub_span)
        # input_po_feature = tf.concat([input_lstm_value, input_sub_feature], axis=-1)
        #
        # po_preds = self.po_classifier(input_po_feature)
        #
        # sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
        # po_preds = tf.transpose(po_preds, perm=[0, 2, 1])
        #
        # return sub_preds, po_preds, mask_value
