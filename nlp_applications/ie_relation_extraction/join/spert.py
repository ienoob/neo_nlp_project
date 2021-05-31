#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:42
    @Author  : jack.li
    @Site    : 
    @File    : sp_ert.py

    实现 Span-based Joint Entity and Relation Extraction with Transformer Pre-training


    参考 https://github.com/lavis-nlp/spert
"""
import json
import pickle
import numpy as np
import tensorflow as tf


def batch_index(first_list, index_list):
    batch_num = first_list.shape[0]
    relation_num = index_list.shape[1]

    new_data = []
    for i in range(batch_num):
        batch_data = []
        for j in range(relation_num):
            batch_data.append(tf.gather(first_list, axis=1, indices=index_list[i][j]))
        new_data.append(batch_data)

    return tf.cast(new_data, dtype=tf.float32)



def build_entity_feature(input_embed, entity_mask, input_size: tf.Tensor):
    inner_batch_num = input_embed.shape[0]
    input_size = tf.reshape(input_size, (inner_batch_num,))
    entity_feature = tf.repeat(input_embed, input_size, axis=0)
    # for i in range(inner_batch_num):
    #     for v in range(input_size[i]):
    #         entity_feature.append(input_embed[i])
    # entity_feature = tf.cast(entity_feature, dtype=tf.float32)
    # m = tf.cast(tf.expand_dims(entity_mask, -1) == 0, tf.float32) * (-1e1)
    # entity_spans_pool = m + entity_feature
    m = tf.expand_dims(entity_mask, -1)
    entity_spans_pool = m * entity_feature
    entity_spans_pool = tf.reduce_max(entity_spans_pool, axis=1)

    return entity_spans_pool


def build_relation_feature(input_embed, entity_spans_pool, input_relation_entity,
                           size_embeddings, rel_mask, input_num, relation_count=0):
    inner_batch_num = input_embed.shape[0]
    input_num = tf.reshape(input_num, (inner_batch_num, ))
    relation_embed_feature = tf.repeat(input_embed, input_num, axis=0)

    def new_func(index_list):
        n_shape = entity_spans_pool.shape[1] * 2
        target_tensor = tf.reshape(tf.gather(entity_spans_pool, index_list), (n_shape,))

        return target_tensor

    def new_func2(index_list):
        n_shape = size_embeddings.shape[1] * 2
        target_tensor = tf.reshape(tf.gather(size_embeddings, index_list), (n_shape,))
        return target_tensor

    relation_entity_feature = tf.map_fn(new_func, input_relation_entity, dtype=tf.float32)
    relation_size_feature = tf.map_fn(new_func2, input_relation_entity, dtype=tf.float32)

    # m = tf.cast(tf.expand_dims(rel_mask, -1) == 0, tf.float32) * (-1e1)
    # relation_spans_pool = m + relation_embed_feature
    m = tf.expand_dims(rel_mask, -1)
    relation_spans_pool = m * relation_embed_feature
    relation_embed = tf.reduce_max(relation_spans_pool, axis=1)
    # relation_entity_featurev = tf.stack(relation_entity_feature)
    # relation_size_featurev = tf.stack(relation_size_feature)
    relation_feature = tf.concat([relation_embed, relation_entity_feature, relation_size_feature], axis=1)

    return relation_feature



class SpERt(tf.keras.models.Model):

    def __init__(self, args):
        super(SpERt, self).__init__()

        self.embed = tf.keras.layers.Embedding(args.char_size, args.embed_size)
        self.size_embed = tf.keras.layers.Embedding(args.size_value, args.size_embed_size)
        # 相比于原始模型，增加双向lstm 层
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.lstm_size, return_sequences=True))
        self.rel_classifier = tf.keras.layers.Dense(args.relation_num, activation="softmax")
        self.entity_classifier = tf.keras.layers.Dense(args.entity_num, activation="softmax")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.rel_dropout = tf.keras.layers.Dropout(0.5)

        # self.cla
        self._relation_types = args.relation_num
        self._entity_types = args.entity_num
        self._max_pairs = args.max_pairs

    def call(self, text_ids, text_contexts, entity_masks, entity_sizes, entity_nums, relations_entity, rel_masks,  relation_nums,
             training=None, mask=None):
        # z这里用普通embed替代bert, 这是为了正常运行，毕竟╮(╯▽╰)╭内存太小了
        h = self.embed(text_ids)
        h = self.bi_lstm(h)

        size_embeddings = self.size_embed(entity_sizes)
        entity_spans_pool = build_entity_feature(h, entity_masks, entity_nums)
        entity_repr = tf.concat([entity_spans_pool, size_embeddings], axis=1)
        entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)
        relation_count = relations_entity.shape[0]
        relation_feature = build_relation_feature(h, entity_spans_pool, relations_entity, size_embeddings, rel_masks, relation_nums, relation_count)
        rel_clf = self.rel_classifier(relation_feature)

        return entity_clf, rel_clf

    def filter_span(self, input_entity_logits, input_entity_start_end, input_max_len, entity_couple_set):
        entity_list = tf.argmax(input_entity_logits, axis=-1)
        entity_list = entity_list.numpy()

        entity_span_list = []
        for i, ix in enumerate(entity_list):
            # entity_span_list.append(i)
            if ix != tf.cast(0, dtype=tf.int32):
                entity_span_list.append(i)
        # entity_span_list = entity_span_list[:10]
        relations_entity = []
        relation_masks = []
        relations_num = 0
        for i in entity_span_list:
            ss, se = input_entity_start_end[i]
            for j in entity_span_list:
                os, oe = input_entity_start_end[j]
                if i == j:
                    continue
                if (entity_list[i], entity_list[j]) not in entity_couple_set:
                    continue
                start = se if se > os else oe
                end = os if se > os else ss
                if start > end:
                    start, end = end, start

                relation_maskv = [1 if ind >= start and ind < end else 0 for ind in range(input_max_len)]
                relation_masks.append(relation_maskv)
                relations_entity.append([i, j])
                relations_num += 1

        return tf.cast(relations_entity, dtype=tf.int32), tf.cast(relation_masks, dtype=tf.float32), tf.cast([[relations_num]], tf.int32)

    def predict(self, text_ids, text_contexts, entity_masks, entity_sizes, entity_nums, input_entity_start_end, input_max_len, entity_couple_set):
        h = self.embed(text_ids)
        h = self.bi_lstm(h)

        size_embeddings = self.size_embed(entity_sizes)
        entity_spans_pool = build_entity_feature(h, entity_masks, entity_nums)
        entity_repr = tf.concat([entity_spans_pool, size_embeddings], axis=1)
        # entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)

        entity_list = tf.argmax(entity_clf, axis=-1)
        entity_list = entity_list.numpy()

        entity_nums_numpy = entity_nums.numpy()
        start = 0
        batch_res = []
        for e_num in entity_nums_numpy:

            rel_res = []
            relations_entity, relation_mask, relation_nums = self.filter_span(entity_clf[start:e_num[0]], input_entity_start_end[start:e_num[0]], input_max_len, entity_couple_set)
            if relations_entity.shape[0]:
                relation_feature = build_relation_feature(h, entity_spans_pool, relations_entity, size_embeddings, relation_mask,
                                                          relation_nums)
                rel_clf = self.rel_classifier(relation_feature)
                rel_clf_argmax = tf.argmax(rel_clf, axis=-1)

                rel_res = [(relations_entity[i].numpy()[0], relations_entity[i].numpy()[1], label) for i, label in enumerate(rel_clf_argmax.numpy())]

                rel_res = [(input_entity_start_end[si][0],  entity_list[si], input_entity_start_end[oi], entity_list[oi], p) for si, oi, p in rel_res if p]

            batch_res.append(rel_res)
            start = e_num[0]
        return batch_res


def test_run_model():
    sample_entity_label = tf.constant([[1, 2, 3]])
    sample_relation_label = tf.constant(([[1]]))
