#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:42
    @Author  : jack.li
    @Site    : 
    @File    : sp_ert.py

    实现 Span-based Joint Entity and Relation Extraction with Transformer Pre-training


"""
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset

data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)


def get_sample_data(batch):

    batch_data = []
    ond_data = dict()
    i = 0
    for doc in data_loader.documents:
        raw_text = doc.train_text
        text_ids = doc.train_text_id

        entity_list = doc.entity_list
        relation_list = doc.relation_list

        entity_span = []
        sub_entity_size = []
        for entity in entity_list:
            sub_entity_size.append(entity.size)

        if i >= batch:
            break

char_size = len(data_loader.char2id)
embed_size = 64
hidden_size = 64
size_embed_size = 64
relation_type = len(data_loader.relation2id)
entity_type = len(data_loader.entity2id)

class BatchDataset(object):
    pass


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
    return tf.stack([first_list[i][index_list[i]] for i in range(batch_num)])


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

        h_large = tf.repeat(tf.expand_dims(h, axis=1), repeat=max(min(relations.shape[1], self._max_pairs)), axis=1)
        rel_clf = tf.zeros([batch_size, relations.shape[1], self._relation_types])
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
        return entity_clf


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

tf.keras.losses.B


spert = SpERt(relation_type, entity_type, 10)

sample_encoding = tf.constant([[1, 2]])
sample_mask = tf.constant([[1, 1]])
sample_entity_mask = tf.constant([[[1, 1], [1, 1], [1, 1]]])
sample_entity_sizes = tf.constant([[2, 3, 2]])
sample_relation = tf.constant([[[0, 1]]])
sample_relation_mask = tf.constant([[1, 1]])

spert(sample_encoding, sample_mask, sample_entity_mask, sample_entity_sizes, sample_relation, sample_relation_mask)







