#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

"""
    使用transformer+tensorflow2 实现 bert + crf
"""
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertMainLayer
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.crf_addons import crf_log_likelihood, viterbi_decode

msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")
bert_model_name = "bert-base-chinese"
class_num = len(msra_data.label2id)


class DataIterator(object):
    def __init__(self, input_loader, input_batch_num):
        self.input_loader = input_loader
        self.input_batch_num = input_batch_num
        self.entity_label2id = {"O": 0}
        self.max_len = 0
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def _transformer2feature(self, sentence, label_list):
        sentence_id = self.tokenizer.encode(sentence)
        label_id = [self.input_loader.label2id[label_i] for label_i in label_list]
        sentence_mask = [1 for _ in sentence_id]

        return {
            "sentence_id": sentence_id,
            "label_id": label_id,
            "sentence_mask": sentence_mask
        }

    def batch_transformer(self, input_batch_data):
        batch_sentence_id = []
        batch_label_id = []
        batch_sentence_mask = []
        for data in input_batch_data:
            batch_sentence_id.append(data["sentence_id"])
            batch_label_id.append(data["label_id"])
            batch_sentence_mask.append(data["sentence_mask"])
        batch_sentence_id = tf.keras.preprocessing.sequence.pad_sequences(batch_sentence_id, padding="post", maxlen=512)
        batch_label_id = tf.keras.preprocessing.sequence.pad_sequences(batch_label_id, padding="post", maxlen=510)
        batch_sentence_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_sentence_mask, padding="post", maxlen=512)
        return {
            "sentence_id": batch_sentence_id,
            "label_id": batch_label_id,
            "sentence_mask": batch_sentence_mask
        }

    def __iter__(self):
        inner_batch_data = []
        for i, sentence in enumerate(self.input_loader.train_sentence_list):
            tf_data = self._transformer2feature(sentence, self.input_loader.train_tag_list[i])
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        if inner_batch_data:
            yield self.batch_transformer(inner_batch_data)


class BertCrfModel(tf.keras.Model):

    def __init__(self, inner_bert_model_name):
        super(BertCrfModel, self).__init__()
        config = BertConfig.from_pretrained(inner_bert_model_name, cache_dir=None)
        print(config)
        self.bert_model = TFBertModel.from_pretrained(inner_bert_model_name)
        self.bert_model = TFBertModel(config)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(class_num, class_num)))
        self.out = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs, training=None, mask=None, labels=None):
        # seg_id = tf.zeros(mask.shape)
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        outputs = self.bert_model(inputs, attention_mask=mask)
        outputs = outputs[0][:, 1:-1, :]

        out_tags = self.out(outputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = crf_log_likelihood(out_tags, label_sequences, text_lens, self.transition_params)

            return out_tags, text_lens, log_likelihood
        else:
            return out_tags, text_lens


bert_crf_model = BertCrfModel(bert_model_name)


optimizer = tf.keras.optimizers.Adam()

def loss_func(input_y, logits):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(input_y, 0))

    mask = tf.cast(mask, dtype=tf.int64)
    lossv = cross_func(input_y, logits, sample_weight=mask)

    return lossv


@tf.function()
def train_step(input_xx, input_yy, input_mask):

    with tf.GradientTape() as tape:
        logits, _, log_likelihood = bert_crf_model(input_xx, True, input_mask, input_yy)
        loss_v1 = -tf.reduce_mean(log_likelihood)
        loss_v2 = loss_func(input_yy, logits)
        loss_v = loss_v1 + loss_v2

    variables = bert_crf_model.trainable_variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v

batch_num = 10
data_iterator = DataIterator(msra_data, batch_num)
epoch = 5
for ep in range(epoch):

    for batch_i, batch in enumerate(data_iterator):
        loss = train_step(batch["sentence_id"], batch["label_id"], batch["sentence_mask"])

        if batch_i % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch_i, loss))

