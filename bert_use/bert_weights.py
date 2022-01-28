#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import collections
import tensorflow as tf
from bert_use.bert import modeling



init_checkpoint = "D:\Work\code\python\\bert_use\pretrained_model\cased_L-12_H-768_A-12\\bert_model.ckpt"
bert_config_path = "D:\Work\code\python\\bert_use\pretrained_model\cased_L-12_H-768_A-12\\bert_config.json"

bert_config = modeling.BertConfig.from_json_file(bert_config_path)
is_training = False
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
segment_ids = tf.constant([[1, 0, 1], [0, 2, 0]])
model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)

tvars = tf.trainable_variables()
init_vars = tf.train.list_variables(init_checkpoint)
initialized_variable_names = {}
name_to_variable = collections.OrderedDict()
for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
        name = m.group(1)
    name_to_variable[name] = var

assignment_map = collections.OrderedDict()
for x in init_vars:
    (name, var) = (x[0], x[1])
    print(name, var)
    if name not in name_to_variable:
        continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

pooled_output = model.get_pooled_output()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(len(model.get_all_encoder_layers()))
    print(model.embedding_table())
    print(model.get_all_encoder_layers()[0].eval())
