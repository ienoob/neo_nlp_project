#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/11 19:36
    @Author  : jack.li
    @Site    : 
    @File    : multi_head.py

    Joint entity recognition and relation extraction as a multi-head selection problem

    参考
    https://github.com/bekou/multihead_joint_entity_relation_extraction
    https://github.com/loujie0822/DeepIE

"""
import jieba
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator

data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)
entity_bio_encoder = {"O": 0}

for i in range(1, len(data_loader.entity2id)):
    entity_bio_encoder["B-{}".format(i)] = len(entity_bio_encoder)
    entity_bio_encoder["I-{}".format(i)] = len(entity_bio_encoder)

batch_num = 5
char_size = len(data_loader.char2id)
word_size = len(data_loader.word2id)
char_embed = 10
word_embed = 10
entity_num = len(entity_bio_encoder)
entity_embed_size = 10
rel_num = len(data_loader.relation2id)
rel_embed_size = 10


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
        entity_label_data = np.zeros(len(text_raw))
        rel_label_data = np.zeros((len(text_raw), len(text_raw)))

        for relation in doc.relation_list:
            sub = relation.sub
            obj = relation.obj

            e_label = "B-{}".format(sub.id)
            entity_label_data[sub.start] = entity_bio_encoder[e_label]
            for iv in range(sub.start+1, sub.end):
                e_label = "I-{}".format(sub.id)
                entity_label_data[iv] = entity_bio_encoder[e_label]

            e_label = "B-{}".format(obj.id)
            entity_label_data[obj.start] = entity_bio_encoder[e_label]
            for iv in range(obj.start + 1, obj.end):
                e_label = "I-{}".format(obj.id)
                entity_label_data[iv] = entity_bio_encoder[e_label]

            rel_label_data[sub.end-1][obj.end-1] = relation.id

        return {
            "char_encode_id": char_encode_id,
            "word_encode_id": word_encode_id,
            "entity_label_data": entity_label_data,
            "rel_label_data": rel_label_data
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_encode_id = []
        batch_word_encode_id = []
        batch_entity_label = []
        batch_rel_label = []

        max_len = 0
        for data in input_batch_data:
            batch_char_encode_id.append(data["char_encode_id"])
            batch_word_encode_id.append(data["word_encode_id"])
            batch_entity_label.append(data["entity_label_data"])

            max_len = max(len(data["char_encode_id"]), max_len)

        for data in input_batch_data:
            rel_label_data = data["rel_label_data"]
            rel_label_data = np.pad(rel_label_data,
                              ((0, max_len - rel_label_data.shape[0]), (0, max_len - rel_label_data.shape[1])),
                              'constant', constant_values=0)
            batch_rel_label.append(rel_label_data)
        batch_char_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_encode_id, maxlen=max_len, padding="post")
        batch_word_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_word_encode_id, maxlen=max_len, padding="post")
        batch_entity_label = tf.keras.preprocessing.sequence.pad_sequences(batch_entity_label, maxlen=max_len, padding="post")
        return {
            "char_encode_id": batch_char_encode_id,
            "word_encode_id": batch_word_encode_id,
            "entity_label_data": batch_entity_label,
            "rel_label_data": tf.cast(batch_rel_label, dtype=tf.int32),
            "max_len": tf.cast(max_len, dtype=tf.int32)
        }


class MultiHeaderModel(tf.keras.Model):

    def __init__(self):
        super(MultiHeaderModel, self).__init__()

        self.char_embed = tf.keras.layers.Embedding(char_size, char_embed)
        self.word_embed = tf.keras.layers.Embedding(word_size, char_embed)
        self.ent_embed = tf.keras.layers.Embedding(entity_num, entity_embed_size)
        # self.rel_embed = tf.keras.layers.Embedding(rel_num, rel_embed_size)

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10, return_sequences=True))

        self.emission = tf.keras.layers.Dense(entity_num)

        self.crf = None
        self.selection_u = tf.keras.layers.Dense(rel_num)
        self.selection_v = tf.keras.layers.Dense(rel_num)
        self.selection_uv = tf.keras.layers.Dense(rel_num)

        self.entity_classifier = tf.keras.layers.Dense(entity_num)
        self.rel_classifier = tf.keras.layers.Dense(rel_num, activation="softmax")


    def call(self, char_ids, word_ids, entity_ids, data_max_len, training=None, mask=None):
        mask_value = tf.not_equal(char_ids, 0)
        char_embed = self.char_embed(char_ids)
        word_embed = self.word_embed(word_ids)

        embed = tf.concat([char_embed, word_embed], axis=-1)
        sent_encoder = self.bi_lstm(embed, mask=mask_value)
        # eimission = self.emission(sent_encoder)
        entity_logits = self.entity_classifier(sent_encoder)

        ent_encoder = self.ent_embed(entity_ids)

        rel_encoder = tf.concat((sent_encoder, ent_encoder), axis=-1)
        B, L, H = rel_encoder.shape
        if L is None:
            L = data_max_len
        u = tf.expand_dims(self.selection_u(rel_encoder), axis=1)
        u = tf.keras.activations.relu(tf.tile(u, multiples=(1, L, 1, 1)))
        v = tf.expand_dims(self.selection_v(rel_encoder), axis=2)
        v = tf.keras.activations.relu(tf.tile(v, multiples=(1, 1, L, 1)))
        uv = self.selection_uv(tf.concat((u, v), axis=-1))
        # print(self.rel_embed.get_weights())
        rel_logits = self.rel_classifier(uv)
        # selection_logits = tf.einsum('bijh,rh->birj', uv, self.rel_embed.get_weights())
        #
        return entity_logits, rel_logits


def test_run_model():
    model = MultiHeaderModel()
    sample_char_id = tf.constant([[1, 2]])
    sample_word_id = tf.constant([[2, 3]])
    sample_entity_id = tf.constant([[1, 2]])
    sample_rel_id = tf.constant([[[1, 2], [2, 1]]])

    o_entity_logits, o_rel_logits = model(sample_char_id, sample_word_id, sample_entity_id)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    print(loss_func(sample_rel_id, o_rel_logits))


# test_run_model()
model = MultiHeaderModel()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()


@tf.function(experimental_relax_shapes=True)
def train_step(input_char_id, input_word_id, input_entity_id, input_rel_id, data_max_len):

    with tf.GradientTape() as tape:
        o_entity_logits, o_rel_logits = model(input_char_id, input_word_id, input_entity_id, data_max_len)
        loss_v = loss_func(input_entity_id, o_entity_logits) + loss_func(input_rel_id, o_rel_logits)

        variables = model.variables
        gradients = tape.gradient(loss_v, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    return loss_v

data_iter = DataIter(data_loader)
epoch = 100
for ep in range(epoch):
    for batch_i, batch_data in enumerate(data_iter.train_iter(batch_num)):

        loss_value = train_step(batch_data["char_encode_id"],
                                batch_data["word_encode_id"],
                                batch_data["entity_label_data"],
                                batch_data["rel_label_data"],
                                batch_data["max_len"])
        if batch_i % 100 == 0:
            print("epoch {0} batch {1} loss value {2}".format(ep, batch_i, loss_value))
    break








