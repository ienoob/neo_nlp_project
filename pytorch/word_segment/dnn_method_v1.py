#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

class MultiTaskModelV2(nn.Module):
    def __init__(self, config):
        super(MultiTaskModelV2, self).__init__()
        self.config = config
        self.electra_model = ElectraModel.from_pretrained(config.bert_model_name)
        self.seg = nn.Linear(config.hidden_size, config.seg_size)

    def forward(self, input_ids, attention_masks, token_type_ids):
        encoder = self.electra_model(input_ids, attention_masks, token_type_ids)
        encoder = encoder[0][:,1:-1,:]

        seg_logits = self.seg(encoder)

        return seg_logits
