#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

"""
    使用transformer+tensorflow2 实现 bert + crf
"""
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from nlp_applications.data_loader import LoadMsraDataV2

msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")

class DataIterator(object):
    def __init__(self, input_loader, input_batch_num):
        self.input_loader = input_loader
        self.input_batch_num = input_batch_num
        self.entity_label2id = {"O": 0}
        self.max_len = 0

        for e_label in self.input_loader.argument_role2id:
            if e_label == "$unk$":
                continue
            self.entity_label2id[e_label+"_B"] = len(self.entity_label2id)
            self.entity_label2id[e_label + "_I"] = len(self.entity_label2id)


class BertCrfModel(tf.keras.Model):

    def __init__(self, bert_model_name):
        super(BertCrfModel, self).__init__()
        self.bert_model = TFBertModel.from_pretrained("bert-base-chinese")

    def call(self, inputs, training=None, mask=None):
        pass


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = TFBertModel.from_pretrained("bert-base-chinese")
text = "[CLS]数据集为IMDB 电影影评，总共有三个数据文件[SEP]"
print(len(text))
encoded_input = tokenizer.encode(text, return_tensors='tf')
print(encoded_input)
output = model(encoded_input)
print(len(output))
print(output[1].shape)
