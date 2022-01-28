#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/22 22:44
    @Author  : jack.li
    @Site    : 
    @File    : transformer.py

"""
import numpy as np
import tensorflow as tf


def position_embed(input_batch, data_maxlen, embed_size):

    positon_embed_out = np.zeros((data_maxlen, embed_size))

    for i in range(data_maxlen):
        for j in range(embed_size):
            if j % 2:
                positon_embed_out[i][j] = np.cos(i/np.power(10000, j/embed_size))
            else:
                positon_embed_out[i][j] = np.sin(i / np.power(10000, j / embed_size))
    positon_embed_out = positon_embed_out[np.newaxis, :]
    positon_embed_out = np.repeat(positon_embed_out, input_batch, axis=0)
    return positon_embed_out


def self_attention(q, k, v, dim_k, mask=None):

    k = tf.transpose(k, [0, 1, 3, 2])
    qk = tf.matmul(q, k)
    qk = tf.cast(qk, tf.float32)
    qk = tf.divide(qk, np.sqrt(dim_k))
    qk = tf.nn.softmax(qk)
    v = tf.cast(v, tf.float32)
    return tf.matmul(qk, v)


sample_k = tf.constant([1]*24, shape=[2, 2, 3, 2])
sample_q = tf.constant([2]*24, shape=[2, 2, 3, 2])
sample_v = tf.constant([3]*24, shape=[2, 2, 3, 2])

# print(self_attention(sample_q, sample_k, sample_v, 2))


class MultiHeader(tf.keras.layers.Layer):

    def __init__(self, embed_size, head_num=8, input_length=128):
        super(MultiHeader, self).__init__()

        self.sub_len = int(embed_size//head_num)
        self.input_length = input_length
        self.head_num = head_num
        self.embed_size = embed_size
        self.q = tf.keras.layers.Dense(embed_size)
        self.k = tf.keras.layers.Dense(embed_size)
        self.v = tf.keras.layers.Dense(embed_size)

        self.feed_forward = tf.keras.layers.Dense(embed_size)

    def call(self, input_q, input_k, input_v, **kwargs):

        batch = input_q.shape[0]
        q = self.q(input_q)
        k = self.k(input_k)
        v = self.v(input_v)

        q = tf.reshape(q, (batch, -1, self.head_num, self.sub_len))
        k = tf.reshape(k, (batch, -1, self.head_num, self.sub_len))
        v = tf.reshape(v, (batch, -1, self.head_num, self.sub_len))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attention_value = self_attention(q, k, v, self.sub_len)
        attention_value_rs = tf.reshape(attention_value, (batch, -1, self.embed_size))

        return attention_value_rs


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, embed_size, seq_num, header_num=8):
        super(TransformerEncoder, self).__init__()

        self.self_attention_layer = MultiHeader(embed_size, head_num=header_num, input_length=seq_num)
        self.normal_layer1 = tf.keras.layers.LayerNormalization()
        self.feedward = tf.keras.layers.Dense(embed_size)
        self.normal_layer2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        inputs_attention = self.self_attention_layer(inputs, inputs, inputs)
        inputs_value = inputs+inputs_attention
        inputs_value = self.normal_layer1(inputs_value)
        inputs_value_feed = self.feedward(inputs_value)
        inputs_value = inputs_value+inputs_value_feed
        inputs_value = self.normal_layer2(inputs_value)

        return inputs_value


class Decoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_size, seq_num, header_num=8):
        super(Decoder, self).__init__()
        self.mask_attention_layer = MultiHeader(embed_size, head_num=header_num, input_length=seq_num)
        self.normal_layer1 = tf.keras.layers.LayerNormalization()
        self.encoder_attention_layer = MultiHeader(embed_size, head_num=header_num, input_length=seq_num)
        self.normal_layer2 = tf.keras.layers.LayerNormalization()
        self.feedward = tf.keras.layers.Dense(embed_size)
        self.normal_layer3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        input_mask_attention = self.mask_attention_layer(inputs)
        input_value = inputs+input_mask_attention
        input_value = self.normal_layer1(input_value)
        input_value_att = self.encoder_attention_layer(input_value)
        input_value = input_value + input_value_att
        input_value = self.normal_layer2(input_value)
        input_value_feed = self.feedward(input_value)
        input_value = input_value + input_value_feed
        input_value = self.normal_layer3(input_value)

        return input_value


# class Attention(tf.keras.layers.Layer):
#     """多头注意力机制
#     """
#     def __init__(self, nb_head, size_per_head, q_in_dim=None, k_in_dim=None, v_in_dim=None, **kwargs):
#
#         super(Attention, self).__init__(**kwargs)
#
#         self.nb_head = nb_head
#         self.size_per_head = size_per_head
#         self.out_dim = nb_head * size_per_head
#         self.multi_head = tf.keras.layers.MultiHeadAttention(nb_head, size_per_head)
#         self.q_in_dim = q_in_dim
#         self.k_in_dim = k_in_dim
#         self.v_in_dim = v_in_dim
#
#         self.q_kernel = self.add_weight(name='q_kernel',
#                                         shape=(self.q_in_dim, self.out_dim),
#                                         initializer='glorot_normal')
#         self.k_kernel = self.add_weight(name='k_kernel',
#                                         shape=(self.k_in_dim, self.out_dim),
#                                         initializer='glorot_normal')
#         self.v_kernel = self.add_weight(name='w_kernel',
#                                         shape=(self.v_in_dim, self.out_dim),
#                                         initializer='glorot_normal')
#
#     # def build(self, input_shape):
#     #     super(Attention, self).build(input_shape)
#     #     q_in_dim = input_shape[0][-1]
#     #     k_in_dim = input_shape[1][-1]
#     #     v_in_dim = input_shape[2][-1]
#
#     def mask(self, x, mask, mode='mul'):
#         if mask is None:
#             return x
#         else:
#             for _ in range(tf.keras.backend.ndim(x) - tf.keras.backend.ndim(mask)):
#                 mask = tf.expand_dims(mask, tf.keras.backend.ndim(mask))
#             if mode == 'mul':
#                 return x * mask
#             else:
#                 return x - (1 - mask) * 1e10
#     def call(self, q, k, v, v_mask=None, q_mask=None):
#         v_mask, q_mask = v_mask, q_mask
#         # 线性变换
#         # print(q.shape, self.q_kernel.shape)
#         qw = tf.matmul(q, self.q_kernel)
#         kw = tf.matmul(k, self.k_kernel)
#         vw = tf.matmul(v, self.v_kernel)
#         # 形状变换
#         qw = tf.reshape(qw, (-1, tf.shape(qw)[1], self.nb_head, self.size_per_head))
#         kw = tf.reshape(kw, (-1, tf.shape(kw)[1], self.nb_head, self.size_per_head))
#         vw = tf.reshape(vw, (-1, tf.shape(vw)[1], self.nb_head, self.size_per_head))
#         # 维度置换
#         qw = tf.keras.backend.permute_dimensions(qw, (0, 2, 1, 3))
#         kw = tf.keras.backend.permute_dimensions(kw, (0, 2, 1, 3))
#         vw = tf.keras.backend.permute_dimensions(vw, (0, 2, 1, 3))
#         # Attention
#         a = tf.matmul(qw, kw, transpose_b=True) / self.size_per_head**0.5
#         a = tf.keras.backend.permute_dimensions(a, (0, 3, 2, 1))
#         a = self.mask(a, v_mask, 'add')
#         a = tf.keras.backend.permute_dimensions(a, (0, 3, 2, 1))
#         a = tf.keras.backend.softmax(a)
#         # 完成输出
#         o =  tf.matmul(a, vw)
#         o = tf.keras.backend.permute_dimensions(o, (0, 2, 1, 3))
#         o = tf.reshape(o, (-1, tf.shape(o)[1], self.out_dim))
#         o = self.mask(o, q_mask, 'mul')
#         return o
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0][0], input_shape[0][1], self.out_dim)