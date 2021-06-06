#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    打算自己构建bert 模型， 可以使用google 训练的参数
"""
import tensorflow as tf
# import tensorflow.compat.v1 as tf
#
# cpkt_file_name = "D:\data\\bert\\uncased_L-2_H-128_A-2\\bert_model.ckpt"
#
# reader = tf.train.NewCheckpointReader(cpkt_file_name)
# for key in sorted(reader.get_variable_to_shape_map()):
#     ts = reader.get_tensor(key)
#
#     print(key, ts.shape)
    # break


class NeoBert(tf.keras.Model):

    def __init__(self):
        super(NeoBert, self).__init__()

        self.embed = tf.keras.layers.Embedding(10, 64)
        self.embed.embeddings_initializer()



