#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/24 22:45
    @Author  : jack.li
    @Site    : 
    @File    : dilated_gated_conv1d.py

"""
import tensorflow as tf
import tensorflow.keras.backend as K


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = tf.keras.layers.Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h
    seq = tf.keras.layers.Lambda(_gate)([seq, h])
    seq = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq