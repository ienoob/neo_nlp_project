#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from nlp_applications.data_loader import LoaderDuie2Dataset, Document
from nlp_applications.utils import load_word_vector
from nlp_applications.ie_relation_extraction.join.pointer_net import PointerNet

data_path = "D:\\data\\关系抽取\\"
word_embed_path = "D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char"
data_loader = LoaderDuie2Dataset(data_path)
word_embed = load_word_vector(word_embed_path)
# word_embed = {}

batch_num = 30
vocab_size = len(data_loader.char2id)
word_size = len(word_embed)
predicate_num = len(data_loader.relation2id)
embed_size = 32
word_embed_size = 300
lstm_size = 32


class DataIterator(object):
    def __init__(self, input_data_loader):
        self.data_loader = input_data_loader
        self.word2id = dict()

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
        text = doc.raw_text
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
        if batch_data:
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


def evaluation(data_iter, input_model):
    predict_res = []
    for i, batch_data in enumerate(data_iter):
        print("batch {} start".format(i))

        out_sub_preds, out_po_preds, mask = input_model.predict(batch_data["encoding"])
        out_sub_value = tf.map_fn(lambda x: 1 if x > 0.5 else 0, out_sub_preds).numpy()
        out_po_value = tf.map_fn(lambda x: 1 if x > 0.5 else 0, out_po_preds).numpy()
        mask_numpy = mask.numpy()

        batch_num = out_sub_preds.shape[0]
        for b, s_pred in enumerate(out_sub_value):
            p_pred = out_po_value[b]

            for j, sv in enumerate(s_pred[0]):
                for k, pv in enumerate(s_pred[1]):
                    pass






def main():
    boundaries = [100000, 110000]
    values = [0.01, 0.001, 0.001]

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    pm_model = PointerNet(vocab_size, embed_size, word_size, word_embed_size, lstm_size, predicate_num)
    data_iter = DataIterator(data_loader)
    #

    optimizer = tf.keras.optimizers.Adam(lr_schedule)


    def loss_func(input_y, logits, input_mask):
        loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # mask = math_ops.not_equal(input_mask, 0)
        # loss_va = loss_fun(input_y, logits)

        loss_va = loss_fun(input_y, logits, sample_weight=input_mask)
        # loss_va = tf.keras.losses.mse(input_y, logits)

        return tf.reduce_mean(loss_va)


    @tf.function(experimental_relax_shapes=True)
    def train_step(input_x, input_sub_span, input_sub_label, input_po_label):
        with tf.GradientTape() as tape:
            sub_logits, po_logits, data_mask = pm_model(input_x, input_sub_span)
            sub_data_mask = tf.repeat()
            lossv = loss_func(input_sub_label, sub_logits, data_mask) + loss_func(input_po_label, po_logits, data_mask)
        variables = pm_model.variables
        gradients = tape.gradient(lossv, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return lossv

    epoch = 100
    model_path = "D:\\tmp\\pointer_net_model\\model"
    for ep in range(epoch):
        for batch_i, b_data in enumerate(data_iter.train_iter(batch_num)):
            loss_value = train_step(b_data["encoding"], b_data["sub_span"],
                                    b_data["sub_label"], b_data["po_label"])

            if batch_i % 100 == 0:
                print("epoch {0} batch {1} loss value is {2}".format(ep, batch_i, loss_value))
                # pm_model.save_weights(model_path, save_format='tf')


    test_batch_num = 1
    # pm_model.load_weights(model_path)
    batch_data_iter = data_iter.dev_iter(test_batch_num)
    submit_res = []
    batch_i = 0
    save_path = "D:\\tmp\submit_data\\duie.json"
    for i, batch_data in enumerate(batch_data_iter):
        print("batch {} start".format(i))
        out_sub_preds, out_po_preds = pm_model.predict(batch_data["encoding"])

        print(out_sub_preds, out_po_preds)

        break


if __name__ == "__main__":
    main()
