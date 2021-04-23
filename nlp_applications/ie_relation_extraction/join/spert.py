#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:42
    @Author  : jack.li
    @Site    : 
    @File    : sp_ert.py

    实现 Span-based Joint Entity and Relation Extraction with Transformer Pre-training


"""
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document

batch_num = 2
data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)

print(len(data_loader.documents))


def convict_data(input_batch_data):
    batch_encodings = []
    batch_context_mask = []
    batch_entity_span = []
    batch_entity_masks = []
    batch_entity_sizes = []
    batch_relations = []
    batch_rel_masks = []
    batch_entity_views = []

    max_len = 0
    for data in input_batch_data:
        batch_encodings.append(data["encoding"])
        batch_context_mask.append(data["context_mask"])
        batch_entity_span.append(data["entity_span"])
        max_len = max(max_len, len(data["encoding"]))

    for data in input_batch_data:

        for entity_mask in data["entity_mask"]:
            batch_entity_masks.append(entity_mask)
        for entity_size in data["entity_size"]:
            batch_entity_sizes.append(entity_size)
        for relation in data["relation_labels"]:
            batch_relations.append(relation)
        for rel_mask in data["relation_masks"]:
            batch_rel_masks.append(rel_mask)


    return {
        "encodings": tf.keras.preprocessing.sequence.pad_sequences(batch_encodings, padding="post"),
        "context_masks": tf.keras.preprocessing.sequence.pad_sequences(batch_context_mask, padding="post"),
        "entity_spans": batch_entity_span,
        "entity_masks": tf.keras.preprocessing.sequence.pad_sequences(batch_context_mask, padding="post"),
        "entity_sizes": batch_entity_sizes,
        "entity_view": batch_entity_views,
        "relations": batch_relations,
        "rel_masks": batch_rel_masks
    }


def sample_single_data(doc: Document):
    text_ids = doc.text_id
    text_len = len(text_ids)

    entity_list = doc.entity_list
    relation_list = doc.relation_list

    entity_span = []
    sub_entity_masks = []
    sub_entity_size = []
    entity_loc_set = set()
    for entity in entity_list:
        entity_mask = [1 if ind >= entity._start and ind < entity._end else 0 for ind in range(text_len)]
        sub_entity_masks.append(entity_mask)
        entity_span.append(entity._id)
        sub_entity_size.append(entity.size)
        entity_loc_set.add((entity._start, entity._end))

    for j in range(text_len - 1):
        for k in range(j + 1, text_len):
            if (j, k) in entity_loc_set:
                continue
            entity_mask = [1 if ind >= j and ind < k else 0 for ind in range(text_len)]
            sub_entity_masks.append(entity_mask)
            entity_span.append(0)
            sub_entity_size.append(k - j)

    entity_d = dict()

    relation_entity_data = dict()
    relation_entity_set = set()
    for rl in relation_list:
        relation_entity_data[(rl._relation_sub._id, rl._relation_obj._id)] = rl._id
        relation_entity_set.add(rl._relation_sub._id)
        entity_d[rl._relation_sub._id] = rl._relation_sub
        relation_entity_set.add(rl._relation_obj._id)
        entity_d[rl._relation_obj._id] = rl._relation_obj

    relation_entity_span = []
    relation_entity_list = list(relation_entity_set)
    relation_label = []
    relation_mask = []
    for i, ei in enumerate(relation_entity_list):
        for j, ej in enumerate(relation_entity_list):
            if i == j:
                continue

            relation_entity_span.append((ei, ej))
            relation_label.append(relation_entity_data.get((ei, ej), 0))

            sub = entity_d[ei]
            obj = entity_d[ej]

            start = sub._end if sub._end > obj._start else obj._end
            end = obj._start if sub._end > obj._start else sub._start

            relation_maskv = [1 if ind >= start and ind < end else 0 for ind in range(text_len)]
            relation_mask.append(relation_maskv)

    return {
        "encoding": text_ids,
        "context_mask": [1 for _ in text_ids],
        "entity_span": entity_span,
        "entity_mask": sub_entity_masks,
        "entity_size": sub_entity_size,
        "relation_entity_spans": tf.cast(relation_entity_span, dtype=tf.int64),
        "relation_labels": tf.cast(relation_label, dtype=tf.int64),
        "relation_masks": tf.cast(relation_mask, dtype=tf.int64),
    }


def get_sample_data(input_batch_num):

    batch_data = []
    for doc in data_loader.documents:
        batch_data.append(sample_single_data(doc))
        if len(batch_data)==input_batch_num:
            yield convict_data(batch_data)
            batch_data = []


char_size = len(data_loader.char2id)
embed_size = 64
hidden_size = 64
size_embed_size = 64
relation_type = len(data_loader.relation2id)
entity_type = len(data_loader.entity2id)


data_batch = get_sample_data(batch_num)

for dt in data_batch:
    print(dt)
    break


def get_token(h, x, token):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    # token_h = h.view(-1, emb_size)
    token_h = tf.reshape(h, (-1, emb_size))
    # tf.conti
    # flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    # token_h = token_h[flat == token, :]

    return token_h


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


class SpERt(tf.keras.models.Model):

    def __init__(self, relation_types, entity_types, max_pairs):
        super(SpERt, self).__init__()

        self.embed = tf.keras.layers.Embedding(char_size, embed_size)
        self.size_embed = tf.keras.layers.Embedding(100, size_embed_size)
        self.rel_classifier = tf.keras.layers.Dense(relation_type)
        self.entity_classifier = tf.keras.layers.Dense(entity_type)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.rel_dropout = tf.keras.layers.Dropout(0.5)

        # self.cla
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

    def call(self, encodings, context_masks, entity_masks, entity_sizes, relations, rel_masks,  training=None, mask=None):
        # z这里用普通embed替代bert, 这是为了正常运行，毕竟╮(╯▽╰)╭内存太小了
        h = self.embed(encodings)

        batch_size = encodings.shape[0]
        size_embeddings = self.size_embed(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(h, entity_masks, size_embeddings)

        h_expand = tf.expand_dims(h, axis=1)
        h_large = tf.repeat(h_expand, repeats=max(min(relations.shape[1], self._max_pairs), 1), axis=1)
        # rel_clf = tf.zeros([batch_size, relations.shape[1], self._relation_types])
        rel_clf = self._classify_relations(entity_spans_pool, size_embeddings,
                                           relations, rel_masks, h_large, 0)
        # for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            # rel_clf = self._classify_relations(entity_spans_pool, size_embeddings,
            #                                             relations, rel_masks, h_large, i)
            # rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
        return entity_clf, rel_clf


    def _classify_entities(self, input_embed, input_entity_masks: tf.Tensor, input_size_embed: tf.Tensor):
        m = tf.cast(tf.expand_dims(input_entity_masks, -1)==0, tf.float32)*(-1e30)
        entity_spans_pool = m+tf.repeat(tf.expand_dims(input_embed, 1), repeats=input_entity_masks.shape[1], axis=1)
        entity_spans_pool = tf.reduce_max(entity_spans_pool, axis=2)

        # entity_ctx = get_token(input_embed, input_encodings)
        entity_repr = tf.concat([entity_spans_pool, input_size_embed], axis=2)
        entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans_pool, size_embeddings, relations, rel_masks, h_large, input_i):
        batch_size = entity_spans_pool.shape[0]

        if relations.shape[1] > self._max_pairs:
            relations = relations[:, input_i:input_i+self._max_pairs, :]
            rel_masks = rel_masks[:, input_i:input_i+self._max_pairs, :]

        entity_pair = batch_index(entity_spans_pool, relations)
        entity_pair = tf.reshape(entity_pair, (batch_size, entity_pair.shape[1], -1))

        entity_size = batch_index(size_embeddings, relations)
        entity_size = tf.reshape(entity_size, (batch_size, entity_size.shape[1], -1))

        m = tf.cast(tf.expand_dims(rel_masks, -1)==0, tf.float32)*(-1e30)
        hm = m + h_large
        hm = tf.reduce_max(hm,axis=2)

        relation_repr = tf.concat([entity_pair, entity_size, hm], axis=2)
        relation_repr = self.rel_dropout(relation_repr)

        relation_clf = self.rel_classifier(relation_repr)

        return relation_clf


    def predict(self):
        pass



spert = SpERt(relation_type, entity_type, 10)

sample_encoding = tf.constant([[1, 2]])
sample_mask = tf.constant([[1, 1]])
sample_entity_mask = tf.constant([[[1, 1], [1, 1], [1, 1]]])
sample_entity_sizes = tf.constant([[2, 3, 2]])
sample_relation = tf.constant([[[0, 1]]])
sample_relation_mask = tf.constant([[1, 1]])

sample_entity_res, sample_relation_res = spert(sample_encoding, sample_mask, sample_entity_mask, sample_entity_sizes, sample_relation, sample_relation_mask)


print(sample_entity_res.shape)
print(sample_relation_res.shape)

sample_entity_label = tf.constant([[1, 2, 3]])
sample_relation_label = tf.constant(([[1]]))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss(sample_entity_label, sample_entity_res))
print(loss(sample_relation_label, sample_relation_res))







