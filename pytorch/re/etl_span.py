#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    文章：Joint Extraction of Entities and Relations Based on a Novel Decomposition 
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.
    """

    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout_rate=0, dropout_output=True, rnn_type=nn.LSTM,
                 concat_layers=False, padding=True):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class SentenceEncoder(nn.Module):
    def __init__(self, config, input_size):
        super(SentenceEncoder, self).__init__()
        rnn_type = nn.LSTM if config.rnn_encoder == 'lstm' else nn.GRU
        self.encoder = StackedBRNN(
            input_size=input_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_rnn_layers,
            dropout_rate=config.dropout,
            dropout_output=True,
            concat_layers=False,
            rnn_type=rnn_type,
            padding=True
        )

    def forward(self, input, mask):
        return self.encoder(input, mask)

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

def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result)).to(data.device)

class EtlSpan(nn.Module):
    
    
    def __init__(self, config, word_emb):
        super(EtlSpan, self).__init__()

        self.char_embed = nn.Embedding(num_embeddings=config.char_size, embedding_dim=config.char_embed_size, padding_idx=0)
        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(word_emb, dtype=torch.float32), freeze=True, padding_idx=0)

        self.word_convert_char = nn.Linear(config.word_embed_size, config.char_embed_size, bias=False)
        self.classes_num = config.rel_num

        self.first_sentence_encoder = SentenceEncoder(config, config.char_embed_size)
        self.encoder_layer = nn.TransformerEncoderLayer(config.hidden_dim*2, nhead=config.encoder_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.layer_norm = ConditionalLayerNorm(config.hidden_dim * 2, eps=1e-12)

        self.po_dense = nn.Linear(config.hidden_dim * 2, self.classes_num * 2)
        self.subject_dense = nn.Linear(config.hidden_dim * 2, 2)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, char_ids, word_ids, subject_ids=None, is_train=True, sentence_lens=[]):
        mask = char_ids != 0
        seq_mask = char_ids.eq(0)
        zero_sign = 1

        char_emb = self.char_embed(char_ids)
        word_emb = self.word_convert_char(self.word_embed(word_ids))
        emb = char_emb + word_emb
        sent_encoder = self.first_sentence_encoder(emb, seq_mask)

        if is_train:
            sub_start_encoder = batch_gather(sent_encoder, subject_ids[:, 0])
            sub_end_encoder = batch_gather(sent_encoder, subject_ids[:, 1])
            subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(sent_encoder, subject)

            context_encoder = context_encoder.transpose(1, 0)
            context_encoder = self.transformer_encoder(context_encoder,
                                                       src_key_padding_mask=seq_mask).transpose(0, 1)

            sub_preds = self.subject_dense(sent_encoder)
            po_preds = self.po_dense(context_encoder).reshape(char_ids.size(0), -1, self.classes_num, 2)
            return sub_preds, po_preds, mask
        else:
            subject_preds = nn.Sigmoid()(self.subject_dense(sent_encoder))
            answer_list = list()
            for iv, sub_pred in enumerate(subject_preds.cpu().numpy()):
                sentence_length = sentence_lens[iv]
                start = np.where(sub_pred[:, 0] > 0.5)[0]
                end = np.where(sub_pred[:, 1] > 0.4)[0]
                subjects = []
                for i in start:
                    j = end[end >= i]
                    if i >= sentence_length:
                        break
                    if len(j) > 0:
                        j = j[0]
                        if j >= sentence_length:
                            continue
                        subjects.append((i, j))
                answer_list.append(subjects)

            sent_encoders, pass_ids, subject_ids, token_type_ids, data_idx = [], [], [], [], []
            for i, subjects in enumerate(answer_list):
                if subjects:
                    pass_tensor = char_ids[i, :].unsqueeze(0).expand(len(subjects), char_ids.size(1))
                    new_sent_encoder = sent_encoder[i, :, :].unsqueeze(0).expand(len(subjects), sent_encoder.size(1),
                                                                                 sent_encoder.size(2))
                    token_type_id = torch.zeros((len(subjects), char_ids.size(1)), dtype=torch.long)
                    for index, (start, end) in enumerate(subjects):
                        token_type_id[index, start:end + 1] = 1

                    pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    sent_encoders.append(new_sent_encoder)
                    token_type_ids.append(token_type_id)
                    data_idx += [i]*len(subjects)

            if len(data_idx) == 0:
                subject_ids = torch.zeros(1, 2).long().to(sent_encoder.device)
                po_tensor = torch.zeros(1, sent_encoder.size(1)).long().to(sent_encoder.device)
                return subject_ids, po_tensor, data_idx

            pass_ids = torch.cat(pass_ids).to(sent_encoder.device)
            sent_encoders = torch.cat(sent_encoders).to(sent_encoder.device)
            subject_ids = torch.cat(subject_ids).to(sent_encoder.device)

            flag = False
            split_heads = 1024

            sent_encoders_ = torch.split(sent_encoders, split_heads, dim=0)
            pass_ids_ = torch.split(pass_ids, split_heads, dim=0)
            subject_encoder_ = torch.split(subject_ids, split_heads, dim=0)
            po_preds = list()
            for i in range(len(subject_encoder_)):
                sent_encoders = sent_encoders_[i]
                pass_ids = pass_ids_[i]
                subject_encoder = subject_encoder_[i]

                if sent_encoders.size(0) == 1:
                    flag = True
                    sent_encoders = sent_encoders.expand(2, sent_encoders.size(1), sent_encoders.size(2))
                    subject_encoder = subject_encoder.expand(2, subject_encoder.size(1))
                    pass_ids = pass_ids.expand(2, pass_ids.size(1))
                sub_start_encoder = batch_gather(sent_encoders, subject_encoder[:, 0])
                sub_end_encoder = batch_gather(sent_encoders, subject_encoder[:, 1])
                subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
                context_encoder = self.layer_norm(sent_encoders, subject)
                context_encoder = self.transformer_encoder(context_encoder.transpose(1, 0),
                                                           src_key_padding_mask=pass_ids.eq(0)).transpose(0, 1)
                po_pred = self.po_dense(context_encoder).reshape(subject_encoder.size(0), -1, self.classes_num, 2)

                if flag:
                    po_pred = po_pred[1, :, :, :].unsqueeze(0)

                po_preds.append(po_pred)

            po_tensor = torch.cat(po_preds)
            po_tensor = nn.Sigmoid()(po_tensor)
            return subject_ids, po_tensor, data_idx







