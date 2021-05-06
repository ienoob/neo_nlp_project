#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/6 21:52
    @Author  : jack.li
    @Site    : 
    @File    : tree_lstm.py

"""
import tensorflow as tf

class TreeLSTM(tf.keras.layers.Layer):

    def __init__(self):
        super(TreeLSTM, self).__init__()

    def call(self, inputs, **kwargs):
        pass