#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/4 21:30
    @Author  : jack.li
    @Site    : 
    @File    : duie2_tplink.py

"""
import jieba
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from nlp_applications.ie_relation_extraction.join.tplink import Tplink

data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)
triple_regularity = data_loader.triple_set

batch_num = 5
char_size = len(data_loader.char2id)
word_size = len(data_loader.word2id)
char_embed = 32
word_embed = 32
entity_embed_size = 32
rel_num = len(data_loader.relation2id)
rel_embed_size = 32
lstm_size = 64

print(data_loader.max_seq_len)
print("rel_num {}".format(rel_num))


class DataIter(BaseDataIterator):

    def __init__(self, input_loader):
        super(DataIter, self).__init__(input_loader)
        self.data_loader = input_loader

    def single_doc_processor(self, doc: Document):
        char_encode_id = doc.text_id
        text_raw = doc.raw_text

        word_encode_id = []
        for tword in jieba.cut(text_raw):
            word_encode_id += [self.data_loader.word2id.get(tword, 1)] * len(tword)
        assert len(word_encode_id) == len(char_encode_id)
        entity_label_data = np.zeros((len(text_raw), len(text_raw)))
        hh_label_data = np.zeros((len(text_raw), len(text_raw)))
        tt_label_data = np.zeros((len(text_raw), len(text_raw)))
        mt_mask = np.ones((len(text_raw), len(text_raw)))

        entity_relation_value = []
        for relation in doc.relation_list:
            sub = relation.sub
            obj = relation.obj

            entity_label_data[sub.start][sub.end-1] = 1
            hh_label_data[sub.start][obj.start] = relation.id
            tt_label_data[sub.end-1][obj.end-1] = relation.id

            entity_relation_value.append((sub.start, sub.end-1, obj.start, obj.end-1, relation.id))

        return {
            "char_encode_id": char_encode_id,
            "word_encode_id": word_encode_id,
            "entity_label_data": entity_label_data,
            "hh_label_data": hh_label_data,
            "tt_label_data": tt_label_data,
            "entity_relation_value": entity_relation_value,
            "text_raw": text_raw,
            "mt_mask": mt_mask
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_encode_id = []
        batch_word_encode_id = []
        batch_entity_label = []
        batch_hh_label_data = []
        batch_tt_label_data = []
        batch_entity_relation_value = []
        batch_text_raw = []
        batch_mt_mask = []

        max_len = 0
        for data in input_batch_data:
            batch_text_raw.append(data["text_raw"])
            batch_char_encode_id.append(data["char_encode_id"])
            batch_word_encode_id.append(data["word_encode_id"])
            batch_entity_relation_value.append(data["entity_relation_value"])

            max_len = max(len(data["char_encode_id"]), max_len)

        for data in input_batch_data:
            entity_label_data = data["entity_label_data"]
            # print(max_len, entity_label_data.shape)
            entity_label_data = np.pad(entity_label_data,
                              ((0,  max_len - entity_label_data.shape[0]), (0, max_len - entity_label_data.shape[1])),
                              'constant', constant_values=0)

            batch_entity_label.append(entity_label_data)
            hh_label_data = data["hh_label_data"]
            hh_label_data = np.pad(hh_label_data,
                                       ((0, max_len - hh_label_data.shape[0]),
                                        (0, max_len - hh_label_data.shape[1])),
                                       'constant', constant_values=0)
            batch_hh_label_data.append(hh_label_data)

            tt_label_data = data["tt_label_data"]
            tt_label_data = np.pad(tt_label_data,
                                   ((0, max_len - tt_label_data.shape[0]),
                                    (0, max_len - tt_label_data.shape[1])),
                                   'constant', constant_values=0)
            batch_tt_label_data.append(tt_label_data)

            mt_mask = data["mt_mask"]
            mt_mask = np.pad(mt_mask,
                                   ((0, max_len - mt_mask.shape[0]),
                                    (0, max_len - mt_mask.shape[1])),
                                   'constant', constant_values=0)
            batch_mt_mask.append(mt_mask)

        batch_char_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_encode_id, maxlen=max_len, padding="post")
        batch_word_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_word_encode_id, maxlen=max_len, padding="post")
        return {
            "char_encode_id": batch_char_encode_id,
            "word_encode_id": batch_word_encode_id,
            "entity_label_data": tf.cast(batch_entity_label, dtype=tf.int32),
            "hh_label_data": tf.cast(batch_hh_label_data, dtype=tf.int32),
            "tt_label_data": tf.cast(batch_tt_label_data, dtype=tf.int32),
            "entity_relation_value": batch_entity_relation_value,
            "text_raw": batch_text_raw,
            "max_len": tf.cast(max_len, dtype=tf.int32),
            "mt_mask": tf.cast(batch_mt_mask, dtype=tf.int32)
        }

    def dev_iter(self, input_batch_num):
        c_batch_data = []
        for doc in self.data_loader.dev_documents:
            c_batch_data.append(self.single_doc_processor(doc))
            if len(c_batch_data) == input_batch_num:
                yield self.padding_batch_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_batch_data(c_batch_data)


def main():
    model = Tplink(char_size, char_embed, word_size, word_embed, lstm_size, rel_num)

    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_funv = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # def loss_funv(input_true, input_logits):
    #     cross_func = tf.keras.losses.SparseCategoricalCrossentropy()
    #     mask = tf.math.logical_not(tf.math.equal(input_true, 0))
    #
    #     mask = tf.cast(mask, dtype=tf.int64)
    #     lossv = cross_func(input_true, input_logits, sample_weight=mask)
    #
    #     return lossv

    optimizer = tf.keras.optimizers.Adam()

    @tf.function(experimental_relax_shapes=True)
    def train_step(input_char_id, input_word_id, input_entity_label, input_hh_label, input_tt_label, input_max_len, input_mt_mask):
        with tf.GradientTape() as tape:
            entity_logits, hh_logits, tt_logits = model(input_char_id, input_word_id, input_max_len)
            loss_v1 = loss_func(input_entity_label, entity_logits)
            loss_v2 = loss_funv(input_hh_label, hh_logits, sample_weight=input_mt_mask)
            loss_v3 = loss_funv(input_tt_label, tt_logits, sample_weight=input_mt_mask)
            loss_v = loss_v1 + loss_v2 + loss_v3

            variables = model.variables
            gradients = tape.gradient(loss_v, variables)
            optimizer.apply_gradients(zip(gradients, variables))
        return loss_v

    data_iter = DataIter(data_loader)
    epoch = 20
    for ep in range(epoch):
        for batch_i, batch_data in enumerate(data_iter.train_iter(batch_num)):
            loss_value = train_step(batch_data["char_encode_id"],
                                    batch_data["word_encode_id"],
                                    batch_data["entity_label_data"],
                                    batch_data["hh_label_data"],
                                    batch_data["tt_label_data"],
                                    batch_data["max_len"],
                                    batch_data["mt_mask"])
            if batch_i % 100 == 0:
                print("epoch {0} batch {1} loss value {2}".format(ep, batch_i, loss_value))


if __name__ == "__main__":
    main()
