#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/7 22:27
    @Author  : jack.li
    @Site    : 
    @File    : match_lstm.py

    Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm and answer pointer.
"""
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuReaderChecklist

train_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：机器阅读理解任务"
data_loader = LoaderDuReaderChecklist(train_path)

vocab_size = len(data_loader.char2id)
embed_size = 64
lstm_size = 64

class DataIter(object):

    def __init__(self, input_data_loader):
        self.data_loader = input_data_loader



class MatchInteraction(tf.keras.layers.Layer):
    def __init__(self):
        super(MatchInteraction, self).__init__()

        self.left_match_rnn = None



class MatchLSTM(tf.keras.Model):

    def __init__(self):
        super(MatchLSTM, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.context_lstm = tf.keras.layers.LSTM(lstm_size)
        self.q_lstm = tf.keras.layers.LSTM(lstm_size)

    def call(self, input_context, input_q, training=None, mask=None):
        context_embed = self.embed(input_context)
        question_embed = self.embed(input_q)

        context_encoder = self.context_lstm(context_embed)
        q_encoder = self.q_lstm(question_embed)





