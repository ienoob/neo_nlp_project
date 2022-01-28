#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    概念解释模型 v1
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader


class ConceptDescDataset(Dataset):
    def __init__(self, sentence_list):
        super(ConceptDescDataset, self).__init__()
        self.sentence_list = sentence_list

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, index):
        return self.sentence_list[index]

    def _create_collate_fn(self, batch_first=False):
        pass


class ConceptDescModelV1(nn.Module):

    def __init__(self, config):
        super(ConceptDescModelV1, self).__init__()
        self.embed = BertModel.from_pretrained(config.pretrain_name)
        self.entity_span = None
        self.desc_span = None

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pass
