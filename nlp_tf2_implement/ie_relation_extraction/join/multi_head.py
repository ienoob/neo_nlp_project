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
import tensorflow as tf
from nlp_applications.ner.crf_addons import crf_log_likelihood, viterbi_decode


class MultiHeaderModel(tf.keras.Model):

    def __init__(self, char_size, char_embed, word_size, entity_num, entity_embed_size, rel_num, rel_embed_size):
        super(MultiHeaderModel, self).__init__()

        self.char_embed = tf.keras.layers.Embedding(char_size, char_embed)
        self.word_embed = tf.keras.layers.Embedding(word_size, char_embed)
        self.ent_embed = tf.keras.layers.Embedding(entity_num, entity_embed_size)
        self.rel_embed = tf.keras.layers.Embedding(rel_num, rel_embed_size)

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))

        # self.emission = tf.keras.layers.Dense(entity_num)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(entity_num, entity_num)))
        self.selection_u = tf.keras.layers.Dense(rel_embed_size)
        self.selection_v = tf.keras.layers.Dense(rel_embed_size)
        self.selection_uv = tf.keras.layers.Dense(rel_embed_size)

        self.entity_classifier = tf.keras.layers.Dense(entity_num, activation="softmax")
        self.rel_classifier = tf.keras.layers.Dense(rel_num, activation="softmax")


    def call(self, char_ids, word_ids, entity_ids=None, data_max_len=None, training=None, mask=None):
        mask_value = tf.math.logical_not(tf.math.equal(char_ids, 0))
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(char_ids, 0), dtype=tf.int32), axis=-1)
        char_embed = self.char_embed(char_ids)
        word_embed = self.word_embed(word_ids)

        embed = tf.concat([char_embed, word_embed], axis=-1)
        sent_encoder = self.bi_lstm(embed, mask=mask_value)
        # eimission = self.emission(sent_encoder)
        entity_logits = self.entity_classifier(sent_encoder)
        if training:
            ent_encoder = self.ent_embed(entity_ids)
            label_sequences = tf.convert_to_tensor(entity_ids, dtype=tf.int32)
            log_likelihood, self.transition_params = crf_log_likelihood(entity_logits, label_sequences, text_lens,
                                                                        self.transition_params)
        else:
            v_entity_ids = []
            for logit, text_len in zip(entity_logits, text_lens):
                viterbi_path, _ = viterbi_decode(logit[:text_len], self.transition_params)
                v_entity_ids.append(viterbi_path)
            entity_ids = tf.keras.preprocessing.sequence.pad_sequences(v_entity_ids, padding='post', dtype=tf.int32)
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
        # selection_logits = tf.einsum('bijh,rh->birj', uv, self.rel_embed.get_weights()[0])
        #
        return entity_logits, rel_logits, mask_value


def test_run_model():
    model = MultiHeaderModel(10, 10, 10, 10, 10, 10, 10)
    sample_char_id = tf.constant([[1, 2]])
    sample_word_id = tf.constant([[2, 3]])
    sample_entity_id = tf.constant([[1, 2]])
    sample_rel_id = tf.constant([[[1, 2], [2, 1]]])

    o_entity_logits, o_rel_logits, _ = model(sample_char_id, sample_word_id, sample_entity_id)
    # loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #
    # print(loss_func(sample_rel_id, o_rel_logits))

test_run_model()
