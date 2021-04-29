#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    文档级别的事件抽取模型
"""

import tensorflow as tf
from typing import List, Callable
import numpy as np
from nlp_applications.data_loader import LoaderBaiduDueeFin, EventDocument, Event, Argument

sample_path = "D:\\data\\篇章级事件抽取\\"
bd_data_loader = LoaderBaiduDueeFin(sample_path)
max_len = 256
print(bd_data_loader.event2id)
print(bd_data_loader.argument_role2id)

for doc in bd_data_loader.document:
    print(doc.title, len(doc.event_list))
    # if len(doc.event_list)==3:
    #     print(doc.text)
    #     for event in doc.event_list:
    #         print(bd_data_loader.id2event[event.id])
    #         for arg in event.arguments:
    #             print(arg.argument, arg.role)


def cut_sentence(input_sentence):
    innser_sentence_list = []
    sentence_len = len(input_sentence)
    cut_char = {"，", " ", "；", "》", "）", "、", ";"}
    indx = 0

    while indx < sentence_len:

        last_ind = min(indx+max_len, sentence_len)
        if last_ind != sentence_len:
            while last_ind > indx:
                if input_sentence[last_ind-1] in cut_char:
                    break
                last_ind -= 1
        if indx == last_ind:
            print(input_sentence)
            raise Exception
        pre_cut = input_sentence[indx:last_ind]
        innser_sentence_list.append(pre_cut)
        indx = last_ind
    for sentence in innser_sentence_list:
        assert len(sentence) <= max_len
    return innser_sentence_list


class DataIter(object):

    def __init__(self, input_loader, input_batch_num):
        self.input_loader = input_loader
        self.input_batch_num = input_batch_num
        self.entity_label = {"O": 0}
        self.max_len = 0

    def _search_index(self, target_word, input_sentence_list):
        out_index = (-1, -1)
        for i, sentence in enumerate(input_sentence_list):
            try:
                index_j = sentence.index(target_word)
                out_index = (i, index_j)
                break
            except ValueError as ve:
                pass
                # print(f'Error Message = {ve}')
        if out_index[0] == -1:
            print(input_sentence_list, target_word)
            raise ValueError

        return out_index

    def _transformer2feature(self, input_doc: EventDocument):
        text = input_doc.text
        title = input_doc.title
        sentences = [title]
        sentences_id = []
        split_char = {"。", "\n"}
        sentence = ""
        for char in text:
            if char in split_char:
                sentence += char
                sentence = sentence.strip()
                if len(sentence) > max_len:
                    tiny_sentence_list = cut_sentence(sentence)
                    sentences += tiny_sentence_list
                elif sentence:
                    sentences.append(sentence)
                sentence = ""
            else:
                sentence += char
        sentence = sentence.strip()
        if sentence:
            if len(sentence) > max_len:
                sentences += cut_sentence(sentence)
            else:
                sentences.append(sentence)
        entity_list = set()
        entity_loc_map = dict()
        for event in input_doc.event_list:
            for arg in event.arguments:
                if arg.is_enum:
                    continue
                row_ind, column_ind_start = self._search_index(arg.argument, sentences)
                entity_list.add((row_ind, column_ind_start, column_ind_start+len(arg.argument), arg.role))

        entity_label = []
        for sentence in sentences:
            sentence_id = [self.input_loader.char2id[char] for char in sentence]

            self.max_len = max(self.max_len, len(sentence_id))
            sentences_id.append(sentence_id)

        return {
            "sentences_id": sentences_id
        }

    def batch_transformer(self, input_batch_data):
        batch_sentences_id = []
        for data in input_batch_data:
            batch_sentences_id.append(data["sentences_id"])

        return {
            "sentences_id": batch_sentences_id
        }

    def __iter__(self):
        inner_batch_data = []
        for doc in self.input_loader.document:
            tf_data = self._transformer2feature(doc)
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        if inner_batch_data:
            yield self.batch_transformer(inner_batch_data)



data_iter = DataIter(bd_data_loader, 2)



print(data_iter.max_len, "hello")

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64


class EventModelDocV1(tf.keras.Model):


    def __init__(self):
        super(EventModelDocV1, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))

    def call(self, inputs, training=None, mask=None):
        batch_num = inputs.shape[0]


for batch_data in data_iter:
    print("=========================")

    break
