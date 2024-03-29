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


class DilatedGatedConv1d(tf.keras.layers.Layer):

    def __init__(self, dim, dilation_rate=1):
        super(DilatedGatedConv1d, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate)
        self.dim = dim

    def call(self, inputs, mask=None, **kwargs):
        seq = inputs
        h = self.conv1(seq)
        def _gate(x):
            dropout_rate = 0.1
            s, ih = x
            g, ih = ih[:, :, :self.dim], ih[:, :, self.dim:]
            g = K.in_train_phase(K.dropout(g, dropout_rate), g)
            g = K.sigmoid(g)
            return g * s + (1 - g) * ih

        seq = _gate([seq, h])
        seq = seq * tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)

        return seq


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = seq.shape[-1]
    h = tf.keras.layers.Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s, ih = x
        g, ih = ih[:, :, :dim], ih[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * ih
    seq = tf.keras.layers.Lambda(_gate)([seq, h])
    seq = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq