#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/2/7 23:20
    @Author  : jack.li
    @Site    : 
    @File    : pcnn.py

"""
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduKg2019RealtionExtraction

input_word_len  = 128
word_embed = 128
input_position_len = 128
position_embed = 128
feature_map = 2
kernel_size = 3
# num_class = 3


class PCNN(tf.keras.Model):

    def __init__(self, word_num, position_num, input_num_class):
        super(PCNN, self).__init__()
        self.word_left_embed = tf.keras.layers.Embedding(word_num, word_embed)
        self.word_right_embed = tf.keras.layers.Embedding(word_num, word_embed)
        self.word_mid_embed = tf.keras.layers.Embedding(word_num, word_embed)

        self.position_left_1_embed = tf.keras.layers.Embedding(position_num, position_embed)
        self.position_left_2_embed = tf.keras.layers.Embedding(position_num, position_embed)
        self.position_right_1_embed = tf.keras.layers.Embedding(position_num, position_embed)
        self.position_right_2_embed = tf.keras.layers.Embedding(position_num, position_embed)
        self.position_mid_1_embed = tf.keras.layers.Embedding(position_num, position_embed)
        self.position_mid_2_embed = tf.keras.layers.Embedding(position_num, position_embed)

        self.left_conv = tf.keras.layers.Conv1D(filters=feature_map, kernel_size=kernel_size)
        self.left_pool = tf.keras.layers.GlobalMaxPool1D()
        self.right_conv = tf.keras.layers.Conv1D(filters=feature_map, kernel_size=kernel_size)
        self.right_pool = tf.keras.layers.GlobalMaxPool1D()
        self.mid_conv = tf.keras.layers.Conv1D(filters=feature_map, kernel_size=kernel_size)
        self.mid_pool = tf.keras.layers.GlobalMaxPool1D()

        self.final_out = tf.keras.layers.Dense(input_num_class, activation="softmax")

    def call(self, word_left_id, word_right_id, word_mid_id, pos_left_1_id, pos_left_2_id, pos_right_1_id,
             pos_right_2_id, pos_mid_1_id, pos_mid_2_id, training=None, mask=None):

        left_word = self.word_left_embed(word_left_id)
        left_pos_1 = self.position_left_1_embed(pos_left_1_id)
        left_pos_2 = self.position_left_2_embed(pos_left_2_id)

        left1 = tf.concat([left_word, left_pos_1, left_pos_2], 2)
        left2 = self.left_conv(left1)
        left = self.left_pool(left2)

        right_word = self.word_right_embed(word_right_id)
        right_pos_1 = self.position_right_1_embed(pos_right_1_id)
        right_pos_2 = self.position_right_2_embed(pos_right_2_id)

        right = tf.concat([right_word, right_pos_1, right_pos_2], 2)
        right = self.right_conv(right)
        right = self.right_pool(right)

        mid_word = self.word_mid_embed(word_mid_id)
        mid_pos_1 = self.position_mid_1_embed(pos_mid_1_id)
        mid_pos_2 = self.position_mid_2_embed(pos_mid_2_id)

        mid = tf.concat([mid_word, mid_pos_1, mid_pos_2], 2)
        mid = self.mid_conv(mid)
        mid = self.mid_pool(mid)

        feature = tf.concat([left, mid, right], 1)

        out = self.final_out(feature)

        return out

# pcnn = PCNN()


data_loader = LoaderBaiduKg2019RealtionExtraction("D:\data\\nlp\百度比赛\Knowledge Extraction")
left_word, right_word, mid_word, left_pos_1, left_pos_2, right_pos_1, right_pos_2, mid_pos_1, mid_pos_2, labels = data_loader.get_train_data()

labels = tf.constant(labels, dtype=tf.int64)
word_num = len(data_loader.word_index)
num_class = len(data_loader.relation_dict)

left_word_seq = tf.keras.preprocessing.sequence.pad_sequences(left_word, maxlen=input_word_len, padding="post")
right_word_seq = tf.keras.preprocessing.sequence.pad_sequences(right_word, maxlen=input_word_len, padding="post")
mid_word_seq = tf.keras.preprocessing.sequence.pad_sequences(mid_word, maxlen=input_word_len, padding="post")

left_pos_1_seq = tf.keras.preprocessing.sequence.pad_sequences(left_pos_1, maxlen=input_position_len, padding="post")
left_pos_2_seq = tf.keras.preprocessing.sequence.pad_sequences(left_pos_2, maxlen=input_position_len, padding="post")
right_pos_1_seq = tf.keras.preprocessing.sequence.pad_sequences(right_pos_1, maxlen=input_position_len, padding="post")
right_pos_2_seq = tf.keras.preprocessing.sequence.pad_sequences(right_pos_2, maxlen=input_position_len, padding="post")
mid_pos_1_seq = tf.keras.preprocessing.sequence.pad_sequences(mid_pos_1, maxlen=input_position_len, padding="post")
mid_pos_2_seq = tf.keras.preprocessing.sequence.pad_sequences(mid_pos_2, maxlen=input_position_len, padding="post")

dataset = tf.data.Dataset.from_tensor_slices((left_word_seq, right_word_seq, mid_word_seq, left_pos_1_seq,
                                              left_pos_2_seq, right_pos_1_seq, right_pos_2_seq, mid_pos_1_seq,
                                              mid_pos_2_seq, labels))

dataset = dataset.shuffle(100).batch(100)

pcnn = PCNN(word_num, 512, num_class)
sample_word_left = tf.constant([[1, 2, 3], [2, 3, 3]])
sample_word_right = tf.constant([[2, 3, 0], [3, 2, 2]])
sample_word_mid = tf.constant([[4, 5, 0], [4, 7, 0]])

sample_pos_left_1 = tf.constant([[0, 1, 2], [4, 5, 6]])
sample_pos_left_2 = tf.constant([[2, 3, 4], [3, 4, 5]])
sample_pos_right_1 = tf.constant([[1, 2, 3], [6, 7, 8]])
sample_pos_right_2 = tf.constant([[4, 5, 6], [1, 2, 3]])
sample_pos_mid_1 = tf.constant([[3, 4, 5], [2, 3, 4]])
sample_pos_mid_2 = tf.constant([[5, 6, 7], [4, 5, 6]])
sample_label = tf.constant([[1], [2]])
p_value = pcnn(sample_word_left, sample_word_right, sample_word_mid, sample_pos_left_1, sample_pos_left_2,
           sample_pos_right_1, sample_pos_right_2, sample_pos_mid_1, sample_pos_mid_2)



loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()


print(loss_func(sample_label, p_value))


@tf.function()
def train_step(left_word, right_word, mid_word, left_pos_1, left_pos_2, right_pos_1, right_pos_2, mid_pos_1, mid_pos_2, input_y):
    with tf.GradientTape() as tape:
        logits = pcnn(left_word, right_word, mid_word, left_pos_1, left_pos_2, right_pos_1, right_pos_2, mid_pos_1, mid_pos_2)
        loss_v = loss_func(input_y, logits)

    variables = pcnn.variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v
#
#
# dataset = None
#


EPOCH = 100
for e in range(EPOCH):

    for i, (left_word, right_word, mid_word, left_pos_1, left_pos_2, right_pos_1, right_pos_2, mid_pos_1, mid_pos_2, train_y) in enumerate(dataset):
        loss = train_step(left_word, right_word, mid_word, left_pos_1, left_pos_2, right_pos_1, right_pos_2, mid_pos_1, mid_pos_2, train_y)

        if i % 100 == 0:
            print("epoch {0} batch {1} loss {2}".format(e, i, loss))
