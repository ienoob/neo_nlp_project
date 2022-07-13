#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/30 12:15
    @Author  : jack.li
    @Site    : 
    @File    : elmo.py

"""
import numpy as np
from tensorflow import keras
import tensorflow as tf

class ELMo(keras.Model):
    def __init__(self, v_dim, emb_dim, units, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.units = units

        # encoder
        self.word_embed = keras.layers.Embedding(
            input_dim=v_dim, output_dim=emb_dim,  # [n_vocab, emb_dim]
            embeddings_initializer=keras.initializers.RandomNormal(0., 0.001),
            mask_zero=True,
        )
        # forward lstm
        self.fs = [keras.layers.LSTM(units, return_sequences=True) for _ in range(n_layers)]
        self.f_logits = keras.layers.Dense(v_dim)
        # backward lstm
        self.bs = [keras.layers.LSTM(units, return_sequences=True, go_backwards=True) for _ in range(n_layers)]
        self.b_logits = keras.layers.Dense(v_dim)

        self.cross_entropy1 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.cross_entropy2 = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(lr)

    def call(self, seqs):
        embedded = self.word_embed(seqs)        # [n, step, dim]
        """
        0123    forward
        1234    forward predict
         1234   backward 
         0123   backward predict
        """
        mask = self.word_embed.compute_mask(seqs)
        fxs, bxs = [embedded[:, :-1]], [embedded[:, 1:]]
        for fl, bl in zip(self.fs, self.bs):
            fx = fl(
                fxs[-1], mask=mask[:, :-1], initial_state=fl.get_initial_state(fxs[-1])
            )           # [n, step-1, dim]
            bx = bl(
                bxs[-1], mask=mask[:, 1:], initial_state=bl.get_initial_state(bxs[-1])
            )  # [n, step-1, dim]
            fxs.append(fx)      # predict 1234
            bxs.append(tf.reverse(bx, axis=[1]))    # predict 0123
        return fxs, bxs

    def step(self, seqs):
        with tf.GradientTape() as tape:
            fxs, bxs = self.call(seqs)
            fo, bo = self.f_logits(fxs[-1]), self.b_logits(bxs[-1])
            loss = (self.cross_entropy1(seqs[:, 1:], fo) + self.cross_entropy2(seqs[:, :-1], bo))/2
        grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss, (fo, bo)

    def get_emb(self, seqs):
        fxs, bxs = self.call(seqs)
        xs = [tf.concat((f[:, :-1, :], b[:, 1:, :]), axis=2).numpy() for f, b in zip(fxs, bxs)]
        for x in xs:
            print("layers shape=", x.shape)
        return np.mean(xs)