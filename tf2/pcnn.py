#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import tensorflow as tf

MAX_LEN = 128
WORD_EMBED_SIZE = 64
WORD_NUMS = 100
feature_map = None
n_class = 10


class PCNN(tf.keras.Model):


    def __init__(self):
        super(PCNN, self).__init__()

        self.word_ids_left = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)
        self.word_ids_mid = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)
        self.word_ids_right = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)

        self.pos_embed_left_1 = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)
        self.pos_embed_left_2 = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)

        self.pos_embed_mid_1 = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)
        self.pos_embed_mid_2 = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)

        self.pos_embed_right_1 = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)
        self.pos_embed_right_2 = tf.keras.layers.Embedding(WORD_NUMS, WORD_EMBED_SIZE)

        self.left_conv = tf.keras.layers.Conv1D(feature_map, kernel_size=3)
        self.left_pool = tf.keras.layers.GlobalMaxPool1D()
        self.mid_conv = tf.keras.layers.Conv1D(feature_map, kernel_size=3)
        self.mid_pool = tf.keras.layers.GlobalMaxPool1D()
        self.right_conv = tf.keras.layers.Conv1D(feature_map, kernel_size=3)
        self.right_pool = tf.keras.layers.GlobalMaxPool1D

        self.out = tf.keras.layers.Dense(n_class, activation="softmax")


    def call(self, word_ids_left, word_ids_mid, word_ids_right, pos_left_1, pos_left_2, pos_mid_1, pos_mid_2, pos_right_1, pos_right_2):

        left_word_embed = self.word_ids_left(word_ids_left)
        mid_word_embed = self.word_ids_mid(word_ids_mid)
        right_word_embed = self.word_ids_right(word_ids_right)

        pos_left_1_embed = self.pos_embed_left_1(pos_left_1)
        pos_left_2_embed = self.pos_embed_left_2(pos_left_2)
        pos_mid_1_embed = self.pos_embed_mid_1(pos_mid_1)
        pos_mid_2_embed = self.pos_embed_mid_2(pos_mid_2)
        pos_right_1_embed = self.pos_embed_right_1(pos_right_1)
        pos_right_2_embed = self.pos_embed_right_2(pos_right_2)

        left_embed = tf.concat([left_word_embed, pos_left_1_embed, pos_left_2_embed], 2)
        mid_embed = tf.concat([mid_word_embed, pos_mid_1_embed, pos_mid_2_embed], 2)
        right_embed = tf.concat([right_word_embed, pos_right_1_embed, pos_right_2_embed], 2)

        left = self.left_conv(left_embed)
        left = self.left_pool(left)

        mid = self.mid_conv(mid_embed)
        mid = self.mid_pool(mid)

        right = self.right_conv(right_embed)
        right = self.right_pool(right)

        final_feature = tf.concat([left, mid, right], 1)

        out = self.out(final_feature)

        return out



pcnn = PCNN()

loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(input_x, target_y):
    with tf.GradientTape() as tape:
        out = pcnn(input_x)

        loss = loss_func(target_y, out)

    variables = pcnn.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))





