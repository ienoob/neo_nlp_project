#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/21 21:49
    @Author  : jack.li
    @Site    : 
    @File    : tensorflow2_bert.py

"""
import tensorflow.compat.v1 as tf
from bert_use.bert_vtf2 import modeling
tf.disable_v2_behavior()

bert_config_file = "D:\data\\bert\\cased_L-12_H-768_A-12\\cased_L-12_H-768_A-12\\bert_config.json"
init_checkpoint = "D:\data\\bert\\cased_L-12_H-768_A-12\\cased_L-12_H-768_A-12\\bert_model.ckpt"

inputs_seq = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_seq") # B * (S+2)
inputs_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_mask") # B * (S+2)
inputs_segment = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_segment")

bert_config = modeling.BertConfig.from_json_file(bert_config_file)
# bert_config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
#     num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

model = modeling.BertModel(config=bert_config, is_training=True,
    input_ids=inputs_seq, input_mask=inputs_mask, token_type_ids=inputs_segment,
                           use_one_hot_embeddings=False)
# init = tf.global_variables_initializer()
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
for var in tvars:
    print(var)
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
# saver = tf.train.Saver()
# ckpt = tf.train.get_checkpoint_state(FLAGS.restore_model)
# saver.restore(sess, ckpt.model_checkpoint_path)
tf_config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())

    # res = sess.run(tvars[10])
    print(tvars[10].get_weights())
    # print(tvars[10])
    # print(res[0][0])

