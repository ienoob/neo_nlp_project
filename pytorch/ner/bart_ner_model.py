#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

res = text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
print(res)
