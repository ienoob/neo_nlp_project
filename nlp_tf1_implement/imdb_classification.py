#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import tensorflow as tf
import numpy as np
import math

imdb_data=np.load("D:\Work\code\python\\bert_use\\imdb.npz",allow_pickle=True)
x_train=imdb_data["x_train"]
y_train=imdb_data["y_train"]
x_test=imdb_data["x_test"]
y_test=imdb_data["y_test"]
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
maxlen=80
batch_size=100
x_train=tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
shuffle_index=np.random.permutation(x_train.shape[0])
x_train=x_train[shuffle_index]
y_train=y_train[shuffle_index]
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#embedding_size=
max_id=0
for arr in x_train:
    for id_ in arr:
        if id_>=max_id:
            max_id=id_
embedding_size=max_id+1
embedding_dim=100
hidden_dim=128


class Dataset:
    def __init__(self, x, y):
        self.x = np.array(x, dtype=np.int32)
        self.y = np.array(y, dtype=np.int32)
        self.sample_nums = x.shape[0]
        self.indicator = 0

    def shuffle_fn(self):
        shuffle_index = np.random.permutation(self.sample_nums)
        self.x = self.x[shuffle_index]
        self.y = self.y[shuffle_index]

    def next_batch(self, batch_size):
        end_indicator = self.indicator + batch_size
        if end_indicator > self.sample_nums:
            self.indicator = 0
            end_indicator = batch_size
            self.shuffle_fn()

        start = self.indicator
        self.indicator += batch_size
        return self.x[start:end_indicator], self.y[start:end_indicator]


def create_model(focal_loss=False):
    inputs = tf.placeholder(tf.int32, shape=[batch_size, maxlen])
    labels = tf.placeholder(tf.int32, shape=[batch_size, ])
    keep_prob = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        embed_matrix = tf.get_variable("embed_arr", shape=[embedding_size, embedding_dim],
                                       dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-1.0, 1.0))
        embeddings = tf.nn.embedding_lookup(params=embed_matrix, ids=inputs)
    scale = 1.0 / math.sqrt(embedding_size + hidden_dim)
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE, initializer=lstm_init):
        lstm_cell1 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, state_is_tuple=True)
        lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, output_keep_prob=keep_prob)
        lstm_cell2 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim, state_is_tuple=True)
        lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, output_keep_prob=keep_prob)

    cell_list = [lstm_cell1, lstm_cell2]
    cells = tf.nn.rnn_cell.MultiRNNCell(cell_list)
    initial_state = cells.zero_state(batch_size, dtype=tf.float32)
    rnn_outputs, _ = tf.nn.dynamic_rnn(cells, inputs=embeddings, initial_state=initial_state)
    # (batch_size,seq_length,hidden_dim)
    output = rnn_outputs[:, -1, :]
    fc_out = tf.layers.dense(inputs=output, units=64, activation=tf.nn.tanh)
    fc_out = tf.nn.dropout(fc_out, keep_prob=keep_prob)
    logits = tf.layers.dense(inputs=fc_out, units=2)
    if focal_loss == False:
        losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=labels))
    else:
        y_pred = tf.nn.softmax(logits)
        losses = focal_loss_fn(y_true=labels, y_pred=y_pred)

    optimizer = tf.train.AdamOptimizer(0.001)
    grads = tf.gradients(losses, tf.trainable_variables())
    grads, _ = tf.clip_by_global_norm(grads, 2.0)
    train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1), dtype=tf.int32), labels),
                dtype=tf.float32))
    return ((inputs, labels, keep_prob), (losses, accuracy, train_op))


def focal_loss_fn(y_true, y_pred, gamma=2, alpha=0.5):
    if y_true.shape != y_pred.shape:
        y_true = tf.one_hot(y_true, 2)
        assert y_true.shape == y_pred.shape

    cross_entroy0 = tf.multiply(y_true, -tf.log(y_pred))
    cross_entroy1 = tf.multiply(tf.subtract(1., y_true), -tf.log(tf.subtract(1., y_pred)))

    fl_0 = tf.power(tf.subtract(1., y_pred), gamma) * alpha * cross_entroy0
    fl_1 = (1 - alpha) * tf.power(y_pred, gamma) * cross_entroy1
    return tf.reduce_mean(tf.add(fl_0, fl_1))


(inputs, labels, keep_prob), (losses, accuracy, train_op) = create_model()
num_batches = x_train.shape[0] // batch_size
train_dataset = Dataset(x_train, y_train)
test_dataset = Dataset(x_test, y_test)


def test_model(sess):
    total_accuracy = 0.0
    total_loss = 0.0
    for i in range(num_batches):
        batch_x, batch_y = test_dataset.next_batch(batch_size)
        print(batch_x.shape, batch_y.shape)
        loss_, acc_val = sess.run([losses, accuracy], feed_dict={inputs: batch_x,
                                                                 labels: batch_y, keep_prob: 1.0})
        total_accuracy += acc_val
        total_loss += loss_
    print("loss is %f and accuracy is %f " % (total_loss / num_batches, total_accuracy / num_batches))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for i in range(num_batches):
            batch_x, batch_y = train_dataset.next_batch(batch_size)
            loss_val, acc, _ = sess.run([losses, accuracy, train_op],
                                        feed_dict={inputs: batch_x, labels: batch_y, keep_prob: 0.3})
            if i % 200 == 0:
                print("Epoch is %d,loss value is %f,accuracy is %f " % (epoch, loss_val, acc))
        print("Currently epoch is ", epoch)
        test_model(sess)

print("*" * 1000)
# (inputs, labels), (losses, accuracy, train_op) = create_model()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(10):
#         for i in range(num_batches):
#             batch_x, batch_y = train_dataset.next_batch(batch_size)
#             loss_val, acc, _ = sess.run([losses, accuracy, train_op],
#                                         feed_dict={inputs: batch_x, labels: batch_y, keep_prob: 0.3})
#             if i % 200 == 0:
#                 print("Epoch is %d,loss value is %f,accuracy is %f " % (epoch, loss_val, acc))
#         print("Currently epoch is ", epoch)
#         test_model(sess)

