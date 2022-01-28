#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/26 22:03
    @Author  : jack.li
    @Site    : 
    @File    : dgcnn.py

"""
import tensorflow as tf
from nlp_applications.layers.dilated_gated_conv1d import dilated_gated_conv1d, DilatedGatedConv1d


class DGCNN(tf.keras.Model):

    def __init__(self):
        super(DGCNN, self).__init__()

        self.embed = tf.keras.layers.Embedding(100, 64)
        self.dgc = DilatedGatedConv1d(64)

    def call(self, inputs, training=None, mask=None):

        x = self.embed(inputs)
        mask = tf.cast(tf.greater(tf.expand_dims(inputs, 2), 0), dtype=tf.float32)
        x = self.dgc([x, mask])

        return x


sample_data = tf.constant([[1, 2, 3, 0, 0]])
sample_mask = tf.constant([[1, 1, 1, 0, 0]])

dgcnn = DGCNN()

out = dgcnn(sample_data, sample_mask)
print(out)