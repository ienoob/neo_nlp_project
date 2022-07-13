#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
from transformers import BertTokenizer, TFBertModel


path = "D:\Work\code\python\\tf2_nlp\\bert_data\cased_L-12_H-768_A-12"
if os.path.isdir(path):
    print(path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."

encoded_input = tokenizer.encode(text, return_tensors='tf')
output = model(encoded_input)

print(output[0].shape)



# model = modeling.BertModel(
#         config=bert_config,
#         is_training=is_training,
#         input_ids=input_ids,
#         input_mask=input_mask,
#         token_type_ids=segment_ids,
#         use_one_hot_embeddings=use_one_hot_embeddings)
