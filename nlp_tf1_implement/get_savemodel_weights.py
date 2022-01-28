#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/5 22:10
    @Author  : jack.li
    @Site    : 
    @File    : get_savemodel_weights.py

"""
import h5py
import tensorflow.compat.v1 as tf
from bert_use.bert_vtf2 import modeling
tf.disable_v2_behavior()

bert_config_file = "D:\data\\bert\\cased_L-12_H-768_A-12\\cased_L-12_H-768_A-12\\bert_config.json"
init_checkpoint = "D:\data\\bert\\cased_L-12_H-768_A-12\\cased_L-12_H-768_A-12\\bert_model.ckpt"

h5FileName = r'net_classification.h5'
reader = tf.train.NewCheckpointReader(init_checkpoint)
f = h5py.File(h5FileName, 'w')
for key in sorted(reader.get_variable_to_shape_map()):
    print(reader.get_tensor(key))
    # f[keyDict] = reader.get_tensor(key)