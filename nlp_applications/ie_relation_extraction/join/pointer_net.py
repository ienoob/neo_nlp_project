#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/11 19:43
    @Author  : jack.li
    @Site    : 
    @File    : pointer_net.py

    Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy

"""
import jieba
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops



class ConditionalLayerNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = tf.ones(hidden_size)
        self.bias = tf.zeros(hidden_size)
        self.variance_epsilon = eps

        self.beta_dense = tf.keras.layers.Dense(hidden_size, bias=False)
        self.gamma_dense = tf.keras.layers.Dense(hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / tf.sqrt(s + self.variance_epsilon)
        return weight * x + bias

def seq_gather(seq, idxs):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    idxs = tf.cast(idxs, tf.int32)
    batch_idxs = tf.range(0, tf.shape(seq)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)
    idxs = tf.concat([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs)


def seq_and_vec(seq, vec):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    vec = tf.expand_dims(vec, 1)
    vec = tf.zeros_like(seq[:, :, :1]) + vec
    return tf.concat([seq, vec], 2)


class PointerNet(tf.keras.models.Model):

    def __init__(self, vocab_size, embed_size, word_size, word_embed_size, lstm_size, predicate_num):
        super(PointerNet, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.word_embed = tf.keras.layers.Embedding(word_size, word_embed_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.sub_classifier = tf.keras.layers.Dense(2, activation="sigmoid")
        self.po_classifier = tf.keras.layers.Dense(predicate_num*2, activation="sigmoid")
        self.normalizer = tf.keras.layers.LayerNormalization()
        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.predicate_num = predicate_num

    def call(self, inputs, word_ids, input_sub_loc=None, training=None, mask=None):
        char_embed = self.embed(inputs)
        word_embed = self.word_embed(word_ids)

        embed = tf.concat([char_embed, word_embed], axis=-1)
        mask_value = math_ops.not_equal(inputs, 0)
        input_lstm_value = self.bi_lstm(embed, mask=mask_value)

        sub_preds = self.sub_classifier(input_lstm_value)
        if not training:
            sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
            mask_value = tf.cast(tf.expand_dims(mask_value, 1), dtype=tf.float32)
            sub_mask_value = tf.repeat(mask_value, 2, axis=1)

            # top_value = tf.math.top_k(sub_preds, 20)
            # print(top_value.values[-1])


            print(sub_preds.shape)
            sub_value = tf.where(tf.greater(sub_preds, 0.5), 1.0, 0.0)
            sub_value = sub_value * sub_mask_value

            print(tf.reduce_max(sub_preds[:, 0, :]))
            print(tf.reduce_max(sub_preds[:, 1, :]))
            print(tf.reduce_sum(sub_value))
            # print(tf.reduce_max(sub_value))
            if tf.reduce_sum(sub_value).numpy() > 100:
                sub_value = tf.where(tf.greater(sub_preds, 0.55), 1.0, 0.0)
                sub_value *= sub_mask_value
                print("new", tf.reduce_sum(sub_value))

            sub_value = sub_value.numpy()
            batch_spo_list = []
            predict_sub_num = 0

            for b, s_pred in enumerate(sub_value):

                spo_list = []
                # print(np.sum(s_pred[0]), b)
                # print(np.sum(s_pred[1]), b)
                for j, sv in enumerate(s_pred[0]):

                    if sv == 0:
                        continue

                    for k, pv in enumerate(s_pred[1]):
                        if k < j:
                            continue
                        if pv == 0:
                            continue
                        # entity_list.append((j, k))
                        predict_sub_num += 1
                        sub_loc_start = tf.cast([[j]], dtype=tf.int32)
                        sub_loc_end = tf.cast([[k]], dtype=tf.int32)

                        sub_start = seq_gather(input_lstm_value[b:b+1,:,:], sub_loc_start)
                        sub_end = seq_gather(input_lstm_value[b:b+1,:,:], sub_loc_end)
                        sub_start_end = tf.concat((sub_start, sub_end), axis=-1)

                        input_po_feature = seq_and_vec(input_lstm_value[b:b+1,:,:], sub_start_end)
                        input_po_feature = self.normalizer(input_po_feature)
                        # input_po_feature = tf.concat([input_lstm_value[b:b+1,:,:], input_sub_feature], axis=-1)

                        po_preds = self.po_classifier(input_po_feature)
                        po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

                        # top_po_value = tf.math.top_k(po_preds, 10).values[-1]
                        # po_preds = tf.where(tf.greater(po_preds, 0.6), 1, 0)
                        po_data_mask = tf.repeat(mask_value, 2 * self.predicate_num, axis=1)
                        po_data_mask = tf.cast(po_data_mask, dtype=tf.float32)

                        po_preds *= po_data_mask
                        po_pred = po_preds.numpy()[0]

                        po_pre_list = []

                        for mi in range(self.predicate_num):
                            if mi == 0:
                                continue
                            po_s_array = po_pred[mi * 2]
                            po_e_array = po_pred[mi * 2 + 1]

                            for mj, pvs in enumerate(po_s_array):
                                if pvs < 0.6:
                                    continue
                                for mk, pve in enumerate(po_e_array):
                                    if mk < mj:
                                        continue
                                    if pve < 0.6:
                                        continue
                                    po_pre_list.append((mj, mk, mi, pvs, pve))
                        po_pre_list.sort(key=lambda x: x[3]+x[4], reverse=True)
                        for mj, mk, mi, _, _ in po_pre_list[:100]:
                            spo_list.append((j, k, mj, mk, mi))
                batch_spo_list.append(spo_list)
            # print(predict_sub_num)
            return batch_spo_list
        else:
            # input_lstm_value = self.dropout(input_lstm_value)
            sub_start = seq_gather(input_lstm_value, input_sub_loc[:, 0:1])
            sub_end = seq_gather(input_lstm_value, input_sub_loc[:, 1:2])
            sub_start_end = tf.concat((sub_start, sub_end), axis=-1)
            input_po_feature = seq_and_vec(input_lstm_value, sub_start_end)
            input_po_feature = self.normalizer(input_po_feature)
            # input_po_feature = tf.concat([input_lstm_value, input_sub_feature], axis=-1)

            po_preds = self.po_classifier(input_po_feature)

            sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
            po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

            return sub_preds, po_preds, mask_value

    # def predict(self, inputs):
        # input_embed = self.embed(inputs)
        # mask_value = math_ops.not_equal(inputs, 0)
        # input_lstm_value = self.bi_lstm(input_embed, mask=mask_value)
        #
        # sub_preds = self.sub_classifier(input_lstm_value)
        #
        # input_sub_span = tf.where(tf.greater(sub_preds, 0.5), 1.0, 0.0)
        # input_sub_span = tf.expand_dims(input_sub_span, axis=-1)
        # input_sub_feature = tf.multiply(input_lstm_value, input_sub_span)
        # input_po_feature = tf.concat([input_lstm_value, input_sub_feature], axis=-1)
        #
        # po_preds = self.po_classifier(input_po_feature)
        #
        # sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
        # po_preds = tf.transpose(po_preds, perm=[0, 2, 1])
        #
        # return sub_preds, po_preds, mask_value
