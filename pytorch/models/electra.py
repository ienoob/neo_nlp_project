#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import torch.nn as nn
from transformers import ElectraModel, AutoModel, ElectraTokenizer


model_path = "D:\data\pre_model\electra_180g_small"

# model = AutoModel.from_pretrained(model_path)
tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-small-discriminator')
model = ElectraModel.from_pretrained("hfl/chinese-electra-small-discriminator")

inputs = tokenizer("数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科，从某种角度看属于形式科学的一种。", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)
