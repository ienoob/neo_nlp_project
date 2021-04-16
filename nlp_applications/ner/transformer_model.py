#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/23 22:59
    @Author  : jack.li
    @Site    : 
    @File    : transformer_model.py

"""
import numpy as np
import tensorflow as tf


vocab_size = 10
embed_size = 64
data_maxlen = 128
class_num = 10

def position_embed(input_batch):

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


def self_attention(q, k, v, dim_k):

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

print(self_attention(sample_q, sample_k, sample_v, 2))


class MultiHeader(tf.keras.layers.Layer):

    def __init__(self, head_num=8, input_length=128):
        super(MultiHeader, self).__init__()

        self.sub_len = int(embed_size//head_num)
        self.embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.input_length = input_length
        self.head_num = head_num
        self.q = tf.keras.layers.Dense(embed_size)
        self.k = tf.keras.layers.Dense(embed_size)
        self.v = tf.keras.layers.Dense(embed_size)

        self.feed_forward = tf.keras.layers.Dense(embed_size)



    def call(self, input_s, **kwargs):

        batch = input_s.shape[0]
        q = self.q(input_s)
        k = self.k(input_s)
        v = self.v(input_s)

        q = tf.reshape(q, (batch, -1, self.head_num, self.sub_len))
        k = tf.reshape(k, (batch, -1, self.head_num, self.sub_len))
        v = tf.reshape(v, (batch, -1, self.head_num, self.sub_len))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        attention_value = self_attention(q, k, v, self.sub_len)
        attention_value_rs = tf.reshape(attention_value, (batch, -1, embed_size))

        return attention_value_rs




# mmh = MultiHeader(2, 4)
#
# sample_input = tf.constant([[1, 2, 3, 4]])
# mmh(sample_input)


class Encoder(tf.keras.layers.Layer):

    def __init__(self, seq_num, header_num=8):
        super(Encoder, self).__init__()

        self.self_attention_layer = MultiHeader(head_num=header_num, input_length=seq_num)
        self.normal_layer1 = tf.keras.layers.LayerNormalization()
        self.feedward = tf.keras.layers.Dense(embed_size)
        self.normal_layer2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        inputs_attention = self.self_attention_layer(inputs)
        inputs_value = inputs+inputs_attention
        inputs_value = self.normal_layer1(inputs_value)
        inputs_value_feed = self.feedward(inputs_value)
        inputs_value = inputs_value+inputs_value_feed
        inputs_value = self.normal_layer2(inputs_value)

        return inputs_value


class Decoder(tf.keras.layers.Layer):

    def __init__(self, seq_num, header_num=8):
        super(Decoder, self).__init__()
        self.mask_attention_layer = MultiHeader(head_num=header_num, input_length=seq_num)
        self.normal_layer1 = tf.keras.layers.LayerNormalization()
        self.encoder_attention_layer = MultiHeader(head_num=header_num, input_length=seq_num)
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



class Transformers(tf.keras.models.Model):

    def __init__(self, encoder_num, decoder_num, seq_num, header_num=8):
        super(Transformers, self).__init__()

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.output_embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.encoders = [Encoder(seq_num, header_num) for _ in range(encoder_num)]
        self.decoders = [Decoder(seq_num, header_num) for _ in range(encoder_num)]

        self.linear = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs):
        pass

