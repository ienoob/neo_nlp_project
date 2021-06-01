#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/23 22:55
    @Author  : jack.li
    @Site    : 
    @File    : pointer_network_model.py

"""
import os
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import extract_entity

msra_data = LoadMsraDataV2("D:\data\\nlp\\命名实体识别\\msra_ner_token_level\\")
char2id = {"pad": 0, "unk": 1}
max_len = -1
msra_train_id = []
for sentence in msra_data.train_sentence_list:
    sentence_id = []
    for s in sentence:
        if s not in char2id:
            char2id[s] = len(char2id)
        sentence_id.append(char2id[s])
    if len(sentence_id) > max_len:
        max_len = len(sentence_id)
    msra_train_id.append(sentence_id)

tag2id = {
    "pad": 0
}
tag_list = msra_data.train_tag_list
msra_start_data = []
msra_end_data = []
msra_span_data = []
for i, tag in enumerate(tag_list):
    tag_start = np.zeros(len(tag))
    tag_end = np.zeros(len(tag))
    et = extract_entity(tag)
    for si, ei, e in et:
        if e not in tag2id:
            tag2id[e] = len(tag2id)
        tag_start[si] = tag2id[e]
        tag_end[ei-1] = tag2id[e]
    msra_start_data.append(tag_start)
    msra_end_data.append(tag_end)

train_data = tf.keras.preprocessing.sequence.pad_sequences(msra_train_id, padding="post", maxlen=max_len)
label_start_data = tf.keras.preprocessing.sequence.pad_sequences(msra_start_data, padding="post", maxlen=max_len)
label_end_data = tf.keras.preprocessing.sequence.pad_sequences(msra_end_data, padding="post", maxlen=max_len)

dataset = tf.data.Dataset.from_tensor_slices((train_data, label_start_data, label_end_data)).shuffle(100).batch(100)




class PointerNetworkModel(tf.keras.Model):
    def __init__(self, char_size, char_embed_size, lstm_embed, ner_num):
        super(PointerNetworkModel, self).__init__()

        self.char_embed = tf.keras.layers.Embedding(char_size, char_embed_size, mask_zero=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_embed, return_sequences=True))
        self.pointer_network_start = tf.keras.layers.Dense(ner_num)
        self.pointer_network_end = tf.keras.layers.Dense(ner_num)

    def call(self, inputs, training=None, mask=None):
        char_value = self.char_embed(inputs)
        mask_value = tf.math.logical_not(tf.math.equal(inputs, 0))
        lstm_value = self.bi_lstm(char_value, mask=mask_value)
        pointer_start_logits = self.pointer_network_start(lstm_value)
        pointer_end_logits = self.pointer_network_end(lstm_value)

        return pointer_start_logits, pointer_end_logits, mask_value


def evaluation(input_sentence_list, input_y, input_model: PointerNetworkModel):
    input_sentence_id = [[char2id.get(c, 1) for c in sentence] for sentence in input_sentence_list]

    eval_batch_num = 100
    batch_value = int(len(input_sentence_list)//eval_batch_num)+1
    predict_res = []
    for b in range(batch_value):
        batch_data = input_sentence_id[b*eval_batch_num:b*eval_batch_num+eval_batch_num]
        if len(batch_data) == 0:
            continue
        batch_data = tf.keras.preprocessing.sequence.pad_sequences(batch_data, padding="post", maxlen=max_len)
        start_logits, end_logits, mask = input_model(batch_data)
        # mask = tf.expand_dims(mask, -1)
        mask = tf.cast(mask, dtype=tf.float32).numpy()
        start_logits_argmax = tf.argmax(start_logits, axis=-1).numpy() * mask
        end_logits_argmax = tf.argmax(end_logits, axis=-1).numpy() * mask

        for i, start_row in enumerate(start_logits_argmax):
            predict_row = []
            t_iv = 0
            for j, start_v in enumerate(start_row):
                if j < t_iv:
                    continue
                for k, end_v in enumerate(end_logits_argmax[i]):
                    if k < j:
                        continue
                    if start_v == end_v:
                        predict_row.append((j, k+1, start_v))
                        t_iv = k+1
                        break
            predict_res.append(predict_row)
    assert len(predict_res) == len(input_y)
    hit_num = 0.0
    predict_num = 0.0
    true_num = 0.0
    for i, predict_row in enumerate(predict_res):
        true_set = set(extract_entity(input_y[i]))
        predict_num += len(predict_row)
        true_num += len(true_set)
        for p in predict_row:
            if p in true_set:
                hit_num += 1

    recall = (hit_num + 1e-8) / (true_num + 1e-3)
    precision = (hit_num + 1e-8) / (predict_num + 1e-3)
    f1_value = 2 * recall * precision / (recall+ precision)
    print("recall {0}, precision {1} f1-value {2}".format(recall, precision, f1_value))


def main():
    model = PointerNetworkModel(len(char2id), 64, 64, len(tag2id))
    optimizer = tf.keras.optimizers.Adam()

    def loss_func(input_y, logits, mask):
        cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        lossv = cross_func(input_y, logits, sample_weight=mask)

        return lossv

    @tf.function()
    def train_step(input_x, input_start_y, input_end_y):
        with tf.GradientTape() as tape:
            start_logits, end_logits, mask = model(input_x, True)
            loss_v = loss_func(input_start_y, start_logits, mask) + loss_func(input_end_y, end_logits, mask)

        variables = model.variables
        gradients = tape.gradient(loss_v, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss_v

    model_data_path = "D:\\tmp\\neo_nlp\\pnm_model"
    # if not os.path.exists(model_data_path):
    #     os.mkdir(model_data_path)
    model.load_weights(model_data_path)
    epoch = 10
    for ep in range(epoch):

        for batch, (train_x, train_start_y, train_end_y) in enumerate(dataset.take(-1)):
            loss_value = train_step(train_x, train_start_y, train_end_y)

            if batch % 100 == 0:
                print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss_value))
                # rnn.save(rnn_data_path)
                model.save_weights(model_data_path, save_format='tf')
    evaluation(msra_data.test_sentence_list, msra_data.test_tag_list, model)


if __name__ == "__main__":
    main()
