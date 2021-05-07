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


class MatchLSTM(tf.keras.Model):

    def __init__(self):
        super(MatchLSTM, self).__init__()



