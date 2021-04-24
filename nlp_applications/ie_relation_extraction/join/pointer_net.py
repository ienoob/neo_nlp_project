#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/11 19:43
    @Author  : jack.li
    @Site    : 
    @File    : pointer_net.py

    Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy

"""
import tensorflow as tf


class PointerNet(tf.keras.models.Model):

    def __init__(self):
        super(PointerNet, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass
