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



class MultiHeaderModel(tf.keras.Model):

    def __init__(self):
        super(MultiHeaderModel, self).__init__()

        self.char_embed = tf.keras.layers.Embedding(char_size, char_embed)
        self.word_embed = tf.keras.layers.Embedding(char_size, char_embed)


