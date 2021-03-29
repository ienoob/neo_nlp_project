#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:42
    @Author  : jack.li
    @Site    : 
    @File    : sp_ert.py

"""
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduKg2019RealtionExtractionV2

data_path = "D:\data\\nlp\百度比赛\Knowledge Extraction"
data_loader = LoaderBaiduKg2019RealtionExtractionV2(data_path)

char_size = len(data_loader.char2id)
embed_size = 64
hidden_size = 64
relation_type = len(data_loader.relation2id)
entity_type = len(data_loader.entity2id)




class SpERt(tf.keras.models.Model):

    def __init__(self, relation_types, entity_types, max_pairs):
        super(SpERt, self).__init__()

        self.embed = tf.keras.layers.Embedding(char_size, embed_size)
        self.rel_classifier = tf.keras.layers.Dense(relation_type)
        self.entity_classifier = tf.keras.layers.Dense(entity_type)

        # self.cla
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs


    def call(self, encodings, context_masks, entity_masks, entity_sizes, relations, rel_masks,  training=None, mask=None):
        # z这里用普通embed替代bert, 这是为了正常运行，毕竟╮(╯▽╰)╭内存太小了
        h = self.embed(encodings)




