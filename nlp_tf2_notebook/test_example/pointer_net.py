#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
import tensorflow as tf

char_num = 10
output_num = 64
word_embed = 32
class_num = 10
lstm_dim = 32


def batch_gather(data, index):
    length = index.shape[0]
    t_index = index.numpy()
    t_data = data.numpy()
    print(t_data.shape)
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return np.array(result)

class ConditionalLayerNorm(tf.keras.layers.Layer):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        # self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.beta_dense = tf.keras.layers.Dense(hidden_size)
        self.gamma_dense = tf.keras.layers.Dense(hidden_size)

    def call(self, x, cond):
        cond = tf.expand_dims(cond, 1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight =  gamma
        bias =  beta

        u = tf.reduce_mean(x, -1, keepdims=True)
        s = tf.reduce_mean(tf.pow((x - u), 2), -1, keepdims=True)
        x = (x - u) / tf.sqrt(s + self.variance_epsilon)
        return weight * x + bias

class PointerNetwork(tf.keras.Model):

    def __init__(self):
        super(PointerNetwork, self).__init__()

        self.char_emb = tf.keras.layers.Embedding(char_num, output_num)
        self.word_emb = tf.keras.layers.Embedding(char_num, word_embed)

        self.word_convert_char = tf.keras.layers.Dense(output_num)

        rnn_num = 2
        rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(rnn_num)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.first_sentence_encoder = tf.keras.layers.RNN(stacked_lstm, return_sequences=True)

        # 这里也是暂时的替换，ε=(´ο｀*)))唉
        self.transformer_encoder = tf.keras.layers.LSTM(lstm_dim, return_sequences=True)

        # 暂时的替换，
        self.layer_norm = ConditionalLayerNorm(128)
        self.po_dense = tf.keras.layers.Dense(2*class_num)
        self.subject_dense = tf.keras.layers.Dense(2)


    def call(self, char_ids, word_ids,  subject_ids, training=None, mask=None):
        seq_mask = tf.equal(char_ids, 0)

        char_embv = self.char_emb(char_ids)
        word_embv = self.word_convert_char(self.word_emb(word_ids))
        emb = char_embv + word_embv
        sent_encoder = self.first_sentence_encoder(emb, mask=seq_mask)

        sub_start_encoder = batch_gather(sent_encoder, subject_ids[:, 0])
        sub_end_encoder = batch_gather(sent_encoder, subject_ids[:, 1])
        subject = tf.concat([sub_start_encoder, sub_end_encoder], 1)
        context_encoder = self.layer_norm(sent_encoder, subject)
        vcontext_encoder = self.transformer_encoder(context_encoder)
        # context_encoder = self.transformer_encoder(tf.transpose(context_encoder, [1, 0, 2]),
        #                                            mask=seq_mask)
        # context_encoder = tf.transpose(context_encoder, [0, 1, 2])

        sub_preds = self.subject_dense(sent_encoder)
        po_preds = self.po_dense(context_encoder)
        po_preds = tf.reshape(po_preds, [char_ids.shape[0], -1, class_num, 2])

        return sub_preds, po_preds


pn = PointerNetwork()

sample_char_id = tf.constant([[1, 2]])
sample_word_id = tf.constant([[2, 3]])
sample_subject_id = tf.constant([[0, 1]])

print(pn(sample_char_id, sample_word_id, sample_subject_id))









