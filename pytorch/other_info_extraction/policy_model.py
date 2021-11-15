#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizer, BertPreTrainedModel, BertModel

def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result)).to(data.device)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.beta_dense = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.gamma_dense = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias


class KeyValueModel(nn.Module):

    def __init__(self, config):
        super(KeyValueModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(config.bert_model_name)
        self.key_dense = nn.Linear(config.hidden_size, 2)

        self.layer_norm = ConditionalLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.value_dense = nn.Linear(config.hidden_size, 2)

    def forward(self, input_x, input_seg_id, input_mask, input_key=None, seq_lens=[], is_eval=False):
        bert_encoder = self.bert_model(input_x, input_seg_id, input_mask)

        if not is_eval:
            sub_start_encoder = batch_gather(bert_encoder, input_key[:, 0])
            sub_end_encoder = batch_gather(bert_encoder, input_key[:, 1])
            subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(bert_encoder, subject)

            key_preds = self.key_dense(bert_encoder)
            value_preds = self.value_dense(context_encoder)
        else:
            subject_preds = nn.Sigmoid()(self.subject_dense(bert_encoder))
            answer_list = list()

            for ii, sub_pred in enumerate(subject_preds.cpu().numpy()):
                start = np.where(sub_pred[:, 0] > 0.6)[0]
                end = np.where(sub_pred[:, 1] > 0.5)[0]
                subjects = []
                for i in start:
                    j = end[end >= i]
                    if i == 0 or i > seq_lens[ii] - 2:
                        continue

                    if len(j) > 0:
                        j = j[0]
                        if j > seq_lens[ii] - 2:
                            continue
                        subjects.append((i, j))

                answer_list.append(subjects)

            sentence_ids = []
            bert_encoders, pass_ids, subject_ids, token_type_ids = [], [], [], []
            for ii, subjects in enumerate(answer_list):
                if subjects:
                    pass_tensor = input_x[ii, :].unsqueeze(0).expand(len(subjects), input_x.size(1))
                    new_bert_encoder = bert_encoder[ii, :, :].unsqueeze(0).expand(len(subjects), bert_encoder.size(1),
                                                                                 bert_encoder.size(2))

                    token_type_id = torch.zeros((len(subjects), input_x.size(1)), dtype=torch.long)
                    for index, (start, end) in enumerate(subjects):
                        token_type_id[index, start:end + 1] = 1
                    sentence_ids.append(ii)
                    pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    bert_encoders.append(new_bert_encoder)
                    token_type_ids.append(token_type_id)

            sentence_ids = torch.cat(sentence_ids).to(bert_encoder.device)
            pass_ids = torch.cat(pass_ids).to(bert_encoder.device)
            bert_encoders = torch.cat(bert_encoders).to(bert_encoder.device)
            subject_ids = torch.cat(subject_ids).to(bert_encoder.device)

            flag = False
            split_heads = 1024

            bert_encoders_ = torch.split(bert_encoders, split_heads, dim=0)
            pass_ids_ = torch.split(pass_ids, split_heads, dim=0)
            subject_encoder_ = torch.split(subject_ids, split_heads, dim=0)

            po_preds = list()
            for i in range(len(bert_encoders_)):
                bert_encoders = bert_encoders_[i]
                pass_ids = pass_ids_[i]
                subject_encoder = subject_encoder_[i]

                if bert_encoders.size(0) == 1:
                    flag = True
                    bert_encoders = bert_encoders.expand(2, bert_encoders.size(1), bert_encoders.size(2))
                    subject_encoder = subject_encoder.expand(2, subject_encoder.size(1))
                sub_start_encoder = batch_gather(bert_encoders, subject_encoder[:, 0])
                sub_end_encoder = batch_gather(bert_encoders, subject_encoder[:, 1])
                subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
                context_encoder = self.LayerNorm(bert_encoders, subject)

                po_pred = self.po_dense(context_encoder).reshape(subject_encoder.size(0), -1, self.classes_num, 2)

                if flag:
                    po_pred = po_pred[1, :, :, :].unsqueeze(0)

                po_preds.append(po_pred)

            po_tensor = torch.cat(po_preds).to(sentence_ids.device)
            po_tensor = nn.Sigmoid()(po_tensor)
            return sentence_ids, subject_ids, po_tensor

import argparse

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--bert_model_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--hidden_size", type=int, default=768, required=False)
    parser.add_argument("--layer_norm_eps", type=float, default=0.001, required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    config = parser.parse_args()

    model = KeyValueModel(config)
    model.to(device)
