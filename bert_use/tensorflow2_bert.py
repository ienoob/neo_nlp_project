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

bert_config_file = "D:\data\\bert\\cased_L-12_H-768_A-12\cased_L-12_H-768_A-12\\bert_config.json"
init_checkpoint = "D:\data\\bert\\cased_L-12_H-768_A-12\cased_L-12_H-768_A-12\\bert_model.ckpt"

input_ids = tf.constant([[31, 51, 99], [15, 5, 0]], name="test1")
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]], name="test2")
token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]], name="test3")

bert_config = modeling.BertConfig.from_json_file(bert_config_file)
# bert_config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
#     num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

model = modeling.BertModel(config=bert_config, is_training=False,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

tvars = tf.trainable_variables()

(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

pooled_output = model.get_pooled_output()
print(model.all_encoder_layers)
# class TF2Bert(tf.keras.models.Model):
#
#     def __init__(self):
#         super(TF2Bert, self).__init__()
#
#         self.bert =