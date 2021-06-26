#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/26 16:05
    @Author  : jack.li
    @Site    : 
    @File    : pipeline_model_v1.py
    pipeline 模式， 先分事件然后在找argument
"""
import tensorflow as tf


class EventTypeClassifier(tf.keras.Model):

    def __init__(self, char_size, char_embed_size, lstm_size, event_num, event_bio_num):
        super(EventTypeClassifier, self).__init__()
        self.embed = tf.keras.layers.Embedding(char_size, char_embed_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.trigger = tf.keras.layers.Dense(event_bio_num, activation="softmax")
        self.classifier = tf.keras.layers.Dense(event_num, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        mask = tf.greater(inputs, 0)
        embed_value = self.embed(inputs)
        lstm_value = self.bi_lstm(embed_value, mask=mask)
        trigger_logits = self.trigger(lstm_value)
        event_logits = self.classifier(lstm_value[:, -1, :])

        return trigger_logits, event_logits, mask


class EventArgument(tf.keras.Model):
    def __init__(self, char_size, char_embed_size, lstm_size, argument_num):
        super(EventArgument, self).__init__()
        self.embed = tf.keras.layers.Embedding(char_size, char_embed_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.argument = tf.keras.layers.Dense(argument_num, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        mask = tf.greater(inputs, 0)
        embed_value = self.embed(inputs)
        lstm_value = self.bi_lstm(embed_value, mask=mask)
        argument_logits = self.argument(lstm_value)
        return argument_logits
