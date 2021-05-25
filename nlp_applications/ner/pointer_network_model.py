#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/23 22:55
    @Author  : jack.li
    @Site    : 
    @File    : pointer_network_model.py

"""
import tensorflow as tf


class PointerNetworkModel(tf.keras.Model):
    def __init__(self):
        super(PointerNetworkModel, self).__init__()
