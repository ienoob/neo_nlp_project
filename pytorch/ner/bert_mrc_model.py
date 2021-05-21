#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    A Unified MRC Framework for Named Entity Recognition
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel, BertConfig
from pytorch.ner.mrc_ner_dataset import MRCNERDataset, collate_to_max_length
from torch.utils.data import DataLoader
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from tokenizers import BertWordPieceTokenizer


class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)

class SingleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)
        return features_output


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

class BertQueryNer(BertPreTrainedModel):

    def __init__(self, config):
        super(BertQueryNer, self).__init__(config)
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout)

        self.hidden_size = config.hidden_size
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits

span_loss_candidates = "all"
loss_type = "bce"
bce_loss = BCEWithLogitsLoss(reduction="none")
dice_smooth = 1e-8
dice_loss = ""

#dice_loss = DiceLoss(with_logits=True, smooth=dice_smooth)

def compute_loss(start_logits, end_logits, span_logits,
                 start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
    batch_size, seq_len = start_logits.size()

    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    if span_loss_candidates == "all":
        # naive mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    else:
        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if span_loss_candidates == "gold":
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    if loss_type == "bce":
        start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
    else:
        start_loss = dice_loss(start_logits, start_labels.float(), start_float_label_mask)
        end_loss = dice_loss(end_logits, end_labels.float(), end_float_label_mask)
        match_loss = dice_loss(span_logits, match_labels.float(), float_match_label_mask)

    return start_loss, end_loss, match_loss

if __name__ == "__main__":
    bert_model_name = "bert-base-chinese"
    bert_path = "D:\data\\bert\\bert-base-chinese"
    json_path = "D:\data\\ner\zh_msra\zh_msra\mrc-ner.train"
    # json_path = "/mnt/mrc/genia/mrc-ner.train"
    is_chinese = True

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,  max_length=512,
                            is_chinese=is_chinese, pad_to_maxlen=False)

    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=collate_to_max_length)

    bert_config = BertQueryNerConfig.from_pretrained(bert_model_name,
                                                     hidden_dropout_prob=0.1,
                                                     attention_probs_dropout_prob=0.1,
                                                     mrc_dropout=0.1)

    model = BertQueryNer.from_pretrained(bert_model_name, config=bert_config)

    optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

    for batch in dataloader:
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
        attention_mask = (tokens != 0).long()

        optimizer.zero_grad()
        start_logits, end_logits, span_logits = model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids)

        start_loss, end_loss, match_loss = compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )
        print(start_loss, end_loss, match_loss)

        optimizer.step()
