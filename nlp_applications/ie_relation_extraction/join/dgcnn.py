#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/26 22:03
    @Author  : jack.li
    @Site    : 
    @File    : dgcnn.py

"""
import tensorflow as tf


class DGCNN(tf.keras.Model):

    def __init__(self):
        super(DGCNN, self).__init__()