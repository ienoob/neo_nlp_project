#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
import tensorflow as tf
from bert_use.bert import modeling

bert_config = modeling.BertConfig.from_json_file("D:\Work\code\python\\bert_use\pretrained_model\cased_L-12_H-768_A-12\\bert_config.json")

input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_input_ids"
        )

input_mask = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_input_mask"
        )

segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="bert_segment_ids"
        )

targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name="Targets"
        )


model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )
tvars = tf.trainable_variables()
init_checkpoint = "D:\Work\code\python\\bert_use\pretrained_model\cased_L-12_H-768_A-12\\bert_model.ckpt"
(assignment_map, initialized_variable_names) = \
    modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
embedded = model.get_pooled_output()
print(embedded.shape)
fc_out = tf.layers.dense(inputs=embedded, units=64, activation=tf.nn.tanh)
# fc_out = tf.nn.dropout(fc_out)
logits = tf.layers.dense(inputs=fc_out, units=2)
print(logits.shape)
losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                               labels=targets))
optimizer = tf.train.AdamOptimizer(0.001)
grads = tf.gradients(losses, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, 2.0)
train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1), dtype=tf.int32), targets),
            dtype=tf.float32))


print(embedded.shape)

n_ntokens = np.array([[1, 2]])
n_tag_ids = np.array([[0, 1]])
n_inputs_ids = np.array([[1, 2]])
n_segment_ids = np.array([[0, 0]])
n_input_mask = np.array([[1, 1]])
n_labels = np.array([0])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {
        input_ids: n_inputs_ids,
        targets: n_labels,
        segment_ids: n_segment_ids,
        input_mask: n_input_mask,
    }

    lossesv = sess.run(
        [losses],
        feed_dict=feed)
    print(lossesv)
    # print(embedding, np.array(embedding).shape)


if __name__ == "__main__":
    pass

