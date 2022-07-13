#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/27 16:22
    @Author  : jack.li
    @Site    : 
    @File    : concept_describe_extract.py

"""
import tensorflow as tf


class ConceptDescribe(tf.keras.Model):

    def __init__(self, char_size, embed_size, lstm_size):
        super(ConceptDescribe, self).__init__()

        self.embed = tf.keras.layers.Embedding(char_size, embed_size)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size))
        self.classifier = tf.keras.layers.Dense(1, activation="sigmoid")
        self.entity_extractor = tf.keras.layers.Dense(3, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        mask = tf.greater(inputs, 0)
        embed = self.embed(inputs)
        bivalue = self.bilstm(embed, mask)

        sentence_end = tf.gather(bivalue, input_end)
        classifier_logits = self.classifier(sentence_end)
        entity_logits = self.entity_extractor(bivalue)

        return classifier_logits, entity_logits






