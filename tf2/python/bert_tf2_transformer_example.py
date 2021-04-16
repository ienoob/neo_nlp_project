#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."

encoded_input = tokenizer.encode(text, return_tensors='tf')
output = model(encoded_input)

print(output[0].shape)
