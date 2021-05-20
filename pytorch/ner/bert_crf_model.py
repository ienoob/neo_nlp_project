#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel
from pytorch.layers.crf import CRF
import torch.autograd as autograd
import torch.optim as optim


class BertNER(nn.Module):
    def __init__(self, gpu_use=False, bertpath=None, label_alphabet_size=2, dropout=0.5):
        super(BertNER, self).__init__()

        self.gpu = gpu_use
        self.data_bertpath = bertpath
        self.bertpath = self.data_bertpath

        char_feature_dim = 768
        print('total char_feature_dim is {}'.format(char_feature_dim))

        self.bert_encoder = BertModel.from_pretrained(self.bertpath)

        self.hidden2tag = nn.Linear(char_feature_dim, label_alphabet_size + 2)
        self.drop = nn.Dropout(p=dropout)

        self.crf = CRF(label_alphabet_size, self.gpu)

        if self.gpu:
            self.bert_encoder = self.bert_encoder.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

    def get_tags(self, batch_bert, bert_mask):
        seg_id = torch.zeros(bert_mask.size()).long().cuda() if self.gpu else torch.zeros(bert_mask.size()).long()
        outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
        outputs = outputs[0][:, 1:-1, :]
        tags = self.hidden2tag(outputs)

        return tags

    def neg_log_likelihood_loss(self, word_inputs, biword_inputs, word_seq_lengths, mask, batch_label, batch_bert,
                                bert_mask):
        tags = self.get_tags(batch_bert, bert_mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, word_inputs, biword_inputs, word_seq_lengths, mask, batch_bert, bert_mask):
        tags = self.get_tags(batch_bert, bert_mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq


input_data = None
bert_model = BertNER(input_data)
parameter = bert_model.parameters()
optimizer = optim.Adamax(parameter)



