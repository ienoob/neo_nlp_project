#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from transformers import BertTokenizer, BertForQuestionAnswering, PreTrainedTokenizerFast
import torch
bert_name = "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
tokenizer = PreTrainedTokenizerFast.from_pretrained(bert_name)
model = BertForQuestionAnswering.from_pretrained(bert_name)

question, text = "企业资质", "（1）科技企业；（2）高新技术企业或高新技术培育企业；（3）苏州市独角兽培育企业；（4）获得各级人才计划资助的领军人才企业。"
inputs = tokenizer(question, text, return_tensors='pt', return_offsets_mapping=True)
# print(inputs)
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
# print(inputs["offset_mapping"])
outputs = model(input_ids=inputs["input_ids"],  token_type_ids=inputs["token_type_ids"], attention_mask=inputs["attention_mask"])
# loss = outputs.loss
# print(outputs)
q_len = len(question)+2
start_scores = torch.sigmoid(outputs.start_logits[0][q_len:])

nd_scores = torch.sigmoid(outputs.end_logits[0][q_len:])

print(start_scores, nd_scores)
start = torch.argmax(start_scores)
end = torch.argmax(nd_scores)
print(len(question), len(text), len(inputs["input_ids"][0]))
print(start, end)
fake_start = inputs["offset_mapping"][0][start+q_len][0]
fake_end = inputs["offset_mapping"][0][end+q_len][1]
print(text[fake_start:fake_end])
print(inputs["input_ids"][0][start:end+1])
