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


class NeoTransformer(tf.keras.layers.Layer):

    def __init__(self, encoder_num, head_num, hidden_size, seq_len):
        super(NeoTransformer, self).__init__()
        self.encoder_layers = [NeoEncoder(head_num, hidden_size, seq_len) for _ in range(encoder_num)]

    def call(self, inputs, *args, **kwargs):
        pass


class NeoEncoder(tf.keras.layers.Layer):

    def __init__(self, head_num, hidden_size, seq_len):
        super(NeoEncoder, self).__init__()
        self.multi_head_layer = NeoMultiHeader(head_num, hidden_size, seq_len)
        self.normal1 = tf.keras.layers.LayerNormalization()
        self.feedforward_layer = tf.keras.layers.Dense(hidden_size)
        self.normal2 = tf.keras.layers.LayerNormalization(0)

    def call(self, inputs, *args, **kwargs):
        mask = kwargs["mask"]
        value = self.multi_head_layer(inputs, mask)
        value1 = self.normal1(value+inputs)
        value2 = self.feedforward_layer(value1)
        value = self.normal2(value1+value2)

        return value


def attention(input_q, input_k, input_v, hidden_size,  mask=None):
    qk = tf.matmul(input_q, input_k, transpose_b=True)/tf.sqrt(hidden_size)
    if mask:
        qk *= (mask*-1e9)
    value = tf.matmul(tf.nn.softmax(qk, axis=-1), input_v)

    return value


class NeoMultiHeader(tf.keras.layers.Layer):

    def __init__(self, head_num, hidden_size, seq_len):
        super(NeoMultiHeader, self).__init__()
        self.head_num = head_num
        self.hidden_size = hidden_size
        assert seq_len % head_num == 0

        self.q = tf.keras.layers.Dense(hidden_size)
        self.k = tf.keras.layers.Dense(hidden_size)
        self.u = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs, *args, **kwargs):
        mask = kwargs["mask"]
        q_value = self.q(inputs)
        k_value = self.k(inputs)
        u_value = self.u(inputs)

        b, l, h = inputs.shape
        q_value = tf.reshape(q_value, (b, self.head_num, l, -1))
        k_value = tf.reshape(k_value, (b, self.head_num, l, -1))
        u_value = tf.reshape(u_value, (b, self.head_num, l, -1))

        value = attention(q_value, k_value, u_value, h, mask)

        return value


class NeoBert(tf.keras.Model):

    def __init__(self, char_size, hidden_size, max_position,  num_hidden_layers, num_attention_heads):
        super(NeoBert, self).__init__()
        self.max_position = max_position
        self.embed = tf.keras.layers.Embedding(char_size, hidden_size)
        self.position_embed = tf.keras.layers.Embedding(max_position, hidden_size)
        self.token_type_embed = tf.keras.layers.Embedding(2, hidden_size)
        self.encoder_layers = [NeoEncoder(num_attention_heads, hidden_size, max_position) for _ in range(num_hidden_layers)]

    def call(self, inputs, input_token_type, training=None, mask=None):
        embed_value = self.embed(inputs)
        batch_num = inputs.shape[0]
        position_input = tf.linspace(tf.zeros(batch_num), tf.ones(batch_num) * (self.max_position-1), self.max_position, name=None)
        position_value = self.position_embed(position_input)
        position_value = tf.transpose(position_value, [1, 0])
        token_value = self.token_type_embed(input_token_type)

        embed = embed_value + position_value + token_value







