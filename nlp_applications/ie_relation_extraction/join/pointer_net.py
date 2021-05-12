#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/11 19:43
    @Author  : jack.li
    @Site    : 
    @File    : pointer_net.py

    Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy

"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from nlp_applications.data_loader import LoaderDuie2Dataset, Document

data_path = "D:\\data\\关系抽取\\"

data_loader = LoaderDuie2Dataset(data_path)

batch_num = 30
vocab_size = len(data_loader.char2id)
predicate_num = len(data_loader.relation2id)
embed_size = 32
lstm_size = 32


class DataIterator(object):
    def __init__(self, input_data_loader):
        self.data_loader = input_data_loader

    def single_doc_processor(self, doc: Document):
        encoding = doc.text_id
        encoding_mask = [1]*len(encoding)
        sub_span = [0]*len(encoding)
        sub_label = np.zeros((2, len(encoding)))
        po_label = np.zeros((2*predicate_num, len(encoding)))
        for relation in doc.relation_list:
            sub_span[relation.sub.start] = 1
            sub_label[0][relation.sub.start] = 1
            sub_span[relation.sub.end-1] = 1
            sub_label[1][relation.sub.end-1] = 1

            pre_type = relation.id
            obj_start = relation.obj.start
            obj_end = relation.obj.end-1
            po_label[pre_type*2][obj_start] = 1
            po_label[pre_type*2+1][obj_end] = 1

        return {"encoding": encoding,
                "encoding_mask": encoding_mask,
                "sub_span": sub_span,
                "sub_label": sub_label,
                "po_label": po_label}

    def single_test_doc_processor(self, doc: Document):
        encoding = doc.text_id
        encoding_mask = [1]*len(encoding)

        return {"encoding": encoding,
                "encoding_mask": encoding_mask,}

    def padding_batch_data(self, input_batch_data):
        batch_encoding = []
        batch_encoding_mask = []
        batch_sub_span = []
        batch_sub_label = []
        batch_po_label = []
        max_len = 0
        for data in input_batch_data:
            batch_encoding.append(data["encoding"])
            batch_encoding_mask.append(data["encoding_mask"])
            batch_sub_span.append(data["sub_span"])

            max_len = max(len(data["encoding"]), max_len)

        for data in input_batch_data:
            sub_label = data["sub_label"]
            sub_label = np.pad(sub_label, ((0, 0), (0, max_len-sub_label.shape[1])), 'constant', constant_values=0)
            po_label = data["po_label"]
            po_label = np.pad(po_label, ((0, 0), (0, max_len - po_label.shape[1])), 'constant', constant_values=0)
            batch_sub_label.append(sub_label)
            batch_po_label.append(po_label)

        return {
            "encoding": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding, padding="post"),
            "encoding_mask": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding_mask, padding="post"),
            "sub_span": tf.keras.preprocessing.sequence.pad_sequences(batch_sub_span, padding="post", dtype="float32"),
            "sub_label": tf.cast(batch_sub_label, dtype=tf.int64),
            "po_label": tf.cast(batch_po_label, dtype=tf.int64)}

    def padding_test_batch_data(self, input_batch_data):
        batch_encoding = []
        batch_encoding_mask = []
        max_len = 0
        for data in input_batch_data:
            batch_encoding.append(data["encoding"])
            batch_encoding_mask.append(data["encoding_mask"])
            max_len = max(len(data["encoding"]), max_len)
        return {
            "encoding": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding, padding="post"),
            "encoding_mask": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding_mask, padding="post")
        }

    def train_iter(self, input_batch_num):
        batch_data = []
        for doc in self.data_loader.documents:
            batch_data.append(self.single_doc_processor(doc))
            if len(batch_data) == input_batch_num:
                yield self.padding_batch_data(batch_data)
                batch_data = []
        if  batch_data:
            yield self.padding_batch_data(batch_data)

    def test_iter(self, input_batch_num):
        batch_data = []
        for doc in self.data_loader.test_documents:
            batch_data.append(self.single_test_doc_processor(doc))
            if len(batch_data) == input_batch_num:
                yield self.padding_test_batch_data(batch_data)
                batch_data = []
        if batch_data:
            yield self.padding_test_batch_data(batch_data)

    def dev_iter(self, input_batch_num):
        batch_data = []
        for doc in self.data_loader.dev_documents:
            batch_data.append(self.single_doc_processor(doc))
            if len(batch_data) == input_batch_num:
                yield self.padding_batch_data(batch_data)
                batch_data = []
        if batch_data:
            yield self.padding_batch_data(batch_data)


class ConditionalLayerNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = tf.ones(hidden_size)
        self.bias = tf.zeros(hidden_size)
        self.variance_epsilon = eps

        self.beta_dense = tf.keras.layers.Dense(hidden_size, bias=False)
        self.gamma_dense = tf.keras.layers.Dense(hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / tf.sqrt(s + self.variance_epsilon)
        return weight * x + bias


class PointerNet(tf.keras.models.Model):

    def __init__(self):
        super(PointerNet, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        # self.word_embed = tf.keras.layers.Embedding.
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.sub_classifier = tf.keras.layers.Dense(2, activation="sigmoid")
        self.po_classifier = tf.keras.layers.Dense(predicate_num*2, activation="sigmoid")

    def call(self, inputs, input_sub_span, training=None, mask=None):
        input_embed = self.embed(inputs)
        mask_value = math_ops.not_equal(inputs, 0)
        input_lstm_value = self.bi_lstm(input_embed, mask=mask_value)

        sub_preds = self.sub_classifier(input_lstm_value)
        input_sub_span = tf.expand_dims(input_sub_span, axis=-1)

        input_sub_feature = tf.multiply(input_lstm_value, input_sub_span)
        input_po_feature = tf.concat([input_lstm_value, input_sub_feature], axis=-1)

        po_preds = self.po_classifier(input_po_feature)

        sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
        po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

        return sub_preds, po_preds

    def predict(self, inputs):
        input_embed = self.embed(inputs)
        mask_value = math_ops.not_equal(inputs, 0)
        input_lstm_value = self.bi_lstm(input_embed, mask=mask_value)

        sub_preds = self.sub_classifier(input_lstm_value)

        input_sub_span = tf.where(tf.logical_and(sub_preds>0.5), tf.ones_like(sub_preds), sub_preds)
        input_sub_span = tf.expand_dims(input_sub_span, axis=-1)
        input_sub_feature = tf.multiply(input_lstm_value, input_sub_span)
        input_po_feature = tf.concat([input_lstm_value, input_sub_feature], axis=-1)

        po_preds = self.po_classifier(input_po_feature)

        sub_preds = tf.transpose(sub_preds, perm=[0, 2, 1])
        po_preds = tf.transpose(po_preds, perm=[0, 2, 1])

        return sub_preds, po_preds


boundaries = [100000, 110000]
values = [0.01, 0.001, 0.001]

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

pm_model = PointerNet()
data_iter = DataIterator(data_loader)
loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(lr_schedule)


def loss_func(input_y, logits, input_mask):
    # mask = math_ops.not_equal(input_mask, 0)
    # loss_va = loss_fun(input_y, logits)
    # loss_va = loss_fun(input_y, logits)
    loss_va = tf.keras.losses.mse(input_y, logits)

    return tf.reduce_mean(loss_va)



@tf.function(experimental_relax_shapes=True)
def train_step(input_x, input_sub_span, input_mask, input_sub_label, input_po_label):
    with tf.GradientTape() as tape:
        sub_logits, po_logits = pm_model(input_x, input_sub_span, mask=input_mask)

        lossv = loss_func(input_sub_label, sub_logits, input_mask) + loss_func(input_po_label, po_logits, input_mask)
    variables = pm_model.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv

epoch = 100
model_path = "D:\\tmp\\pointer_net_model\\model"
for ep in range(epoch):
    for batch_i, b_data in enumerate(data_iter.train_iter(batch_num)):
        loss_value = train_step(b_data["encoding"], b_data["sub_span"], b_data["encoding_mask"],
                                b_data["sub_label"], b_data["po_label"])

        if batch_i % 100 == 0:
            print("epoch {0} batch {1} loss value is {2}".format(ep, batch_i, loss_value))
            pm_model.save_weights(model_path, save_format='tf')



test_batch_num = 1
pm_model.load_weights(model_path)
batch_data_iter = data_iter.dev_iter(test_batch_num)
submit_res = []
batch_i = 0
save_path = "D:\\tmp\submit_data\\duie.json"
for i, batch_data in enumerate(batch_data_iter):
    print("batch {} start".format(i))
    out_sub_preds, out_po_preds = pm_model.predict(batch_data["encoding"])

    print(out_sub_preds, out_po_preds)

    break
