#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/8/7 11:41
    @Author  : jack.li
    @Site    : 
    @File    : pn_dgcnn.py

    dgcnn 指针网络

"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from nlp_applications.layers.dilated_gated_conv1d import DilatedGatedConv1d

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


def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size,1,1,seq_len)

class DgCnnPN(tf.keras.Model):

    def __init__(self, config):
        super(DgCnnPN, self).__init__()
        self.embed = tf.keras.layers.Embedding(config.char_size, config.char_embed_size)
        self.word_embed = tf.keras.layers.Embedding(config.word_size, config.word_embed_size)
        self.position_embed = tf.keras.layers.Embedding(config.max_len, config.pos_embed_size)
        self.dgc1 = DilatedGatedConv1d(config.dgcnn_size)
        self.dgc2 = DilatedGatedConv1d(config.dgcnn_size, 2)
        self.dgc3 = DilatedGatedConv1d(config.dgcnn_size, 5)
        self.dgc4 = DilatedGatedConv1d(config.dgcnn_size)
        self.dgc5 = DilatedGatedConv1d(config.dgcnn_size, 2)
        self.dgc6 = DilatedGatedConv1d(config.dgcnn_size, 5)
        self.dgc7 = DilatedGatedConv1d(config.dgcnn_size)
        self.dgc8 = DilatedGatedConv1d(config.dgcnn_size, 2)
        self.dgc9 = DilatedGatedConv1d(config.dgcnn_size, 5)
        self.dgc10 = DilatedGatedConv1d(config.dgcnn_size)
        self.dgc11 = DilatedGatedConv1d(config.dgcnn_size)
        self.dgc12 = DilatedGatedConv1d(config.dgcnn_size)
        self.attention = tf.keras.layers.MultiHeadAttention(8, 16)
        self.conv1 = tf.keras.layers.Conv1D(config.char_embed_size, 3,  activation='relu', padding='same')
        self.po_attention = tf.keras.layers.MultiHeadAttention(8, 24)
        self.conv2= tf.keras.layers.Conv1D(config.dgcnn_size, 3, activation='relu', padding='same')
        self.predicate_num = config.predicate_num
        self.sub_classifier = tf.keras.layers.Dense(2, activation="sigmoid")
        self.po_classifier = tf.keras.layers.Dense(config.predicate_num * 2, activation="sigmoid")

    def call(self, inputs, inputs_word=None, inputs_position=None, training=None, mask=None, input_sub_loc=None):
        char_embed = self.embed(inputs)
        word_embed = self.word_embed(inputs_word)
        pos_embed = self.position_embed(inputs_position)
        # print(char_embed.shape)
        # print(pos_embed.shape)

        embed = tf.concat([char_embed, word_embed], axis=-1) + pos_embed
        mask_value = math_ops.not_equal(inputs, 0)
        dgc_value = self.dgc1(embed, mask=mask_value)
        dgc_value = self.dgc2(dgc_value, mask=mask_value)
        dgc_value = self.dgc3(dgc_value, mask=mask_value)
        dgc_value = self.dgc4(dgc_value, mask=mask_value)
        dgc_value = self.dgc5(dgc_value, mask=mask_value)
        dgc_value = self.dgc6(dgc_value, mask=mask_value)
        dgc_value = self.dgc7(dgc_value, mask=mask_value)
        dgc_value = self.dgc8(dgc_value, mask=mask_value)
        dgc_value = self.dgc9(dgc_value, mask=mask_value)
        dgc_value = self.dgc10(dgc_value, mask=mask_value)
        dgc_value = self.dgc11(dgc_value, mask=mask_value)
        dgc_value = self.dgc12(dgc_value, mask=mask_value)

        # att_mask = tf.cast(tf.equal(inputs, 0), dtype=tf.float32) * -1e9
        # att_mask = tf.repeat(tf.expand_dims(att_mask, axis=-1), 128, axis=-1)
        att_mask = tf.expand_dims(mask_value, axis=1)
        att_dgc = self.attention(dgc_value, dgc_value, dgc_value, att_mask)
        # print(att.shape)
        att_dgc = tf.concat([dgc_value, att_dgc], axis=-1)
        att_dgc = self.conv1(att_dgc)
        sub_preds = self.sub_classifier(att_dgc)
        # tf.keras.layers.Concatenate()

        if not training:
            sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
            mask_value = tf.cast(tf.expand_dims(mask_value, 1), dtype=tf.float32)
            sub_mask_value = tf.repeat(mask_value, 2, axis=1)

            # top_value = tf.math.top_k(sub_preds, 20)
            # print(top_value.values[-1])

            sub_value = tf.where(tf.greater(sub_preds, 0.5), 1.0, 0.0)
            sub_value = sub_value * sub_mask_value

            sub_value = sub_value.numpy()
            batch_spo_list = []
            batch_entity_list = []
            predict_sub_num = 0

            for b, s_pred in enumerate(sub_value):

                spo_list = []
                entity_list = []
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
                        entity_list.append((j, k))
                        predict_sub_num += 1
                        sub_loc_start = tf.cast([[j]], dtype=tf.int32)
                        sub_loc_end = tf.cast([[k]], dtype=tf.int32)

                        sub_start = seq_gather(att_dgc[b:b + 1, :, :], sub_loc_start)
                        sub_end = seq_gather(att_dgc[b:b + 1, :, :], sub_loc_end)
                        sub_start_end = tf.concat((sub_start, sub_end), axis=-1)

                        input_po_feature = seq_and_vec(att_dgc[b:b + 1, :, :], sub_start_end)
                        # print(input_po_feature.shape, "po_feautre")
                        att_mask_v = att_mask[b:b+1]
                        input_po_feature_att = self.po_attention(input_po_feature, input_po_feature, input_po_feature,
                                                             att_mask_v)
                        input_po_feature = tf.concat([input_po_feature_att, input_po_feature], axis=-1)
                        input_po_feature = self.conv2(input_po_feature)

                        po_preds = self.po_classifier(input_po_feature)
                        po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

                        # top_po_value = tf.math.top_k(po_preds, 10).values[-1]
                        # po_preds = tf.where(tf.greater(po_preds, 0.6), 1, 0)
                        po_data_mask = tf.repeat(mask_value, 2 * self.predicate_num, axis=1)
                        po_data_mask = tf.cast(po_data_mask, dtype=tf.float32)

                        po_preds *= po_data_mask
                        po_pred = po_preds.numpy()[0]

                        for mi in range(self.predicate_num):
                            if mi == 0:
                                continue
                            po_s_array = po_pred[mi * 2]
                            po_e_array = po_pred[mi * 2 + 1]

                            for mj, pvs in enumerate(po_s_array):
                                if pvs < 0.5:
                                    continue
                                for mk, pve in enumerate(po_e_array):
                                    if mk < mj:
                                        continue
                                    if pve < 0.5:
                                        continue
                                    # po_pre_list.append((mj, mk, mi, pvs, pve))
                                    spo_list.append((j, k, mj, mk, mi))

                                    break
                        break
                        # po_pre_list.sort(key=lambda x: x[3] + x[4], reverse=True)
                        # for mj, mk, mi, _, _ in po_pre_list[:10]:
                batch_entity_list.append(entity_list)
                batch_spo_list.append(spo_list)
            # print(predict_sub_num)
            return batch_spo_list, batch_entity_list
        else:
            # input_lstm_value = self.dropout(input_lstm_value)
            sub_start = seq_gather(att_dgc, input_sub_loc[:, 0:1])
            sub_end = seq_gather(att_dgc, input_sub_loc[:, 1:2])
            sub_start_end = tf.concat((sub_start, sub_end), axis=-1)
            input_po_feature = seq_and_vec(att_dgc, sub_start_end)
            # print(input_po_feature.shape)
            input_po_feature_att = self.po_attention(input_po_feature, input_po_feature, input_po_feature, att_mask)
            # input_po_feature = self.normalizer(input_po_feature)
            input_po_feature = tf.concat([input_po_feature_att, input_po_feature], axis=-1)
            input_po_feature = self.conv2(input_po_feature)
            po_preds = self.po_classifier(input_po_feature)

            sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
            po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

            return sub_preds, po_preds, mask_value






