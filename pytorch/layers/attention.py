#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import torch.nn as nn
import torch.nn.functional as F

# luong 2015
class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, dropout_p=0):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        # hc: [hidden, context]
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        # s: softmax
        self.Ws = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        gru_out, hidden = self.gru(embedded, hidden)

        attn_prod = torch.mm(self.attn(hidden)[0], encoder_outputs.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_outputs)

        # hc: [hidden: context]
        hc = torch.cat([hidden[0], context], dim=1)
        out_hc = F.tanh(self.Whc(hc))
        output = F.log_softmax(self.Ws(out_hc), dim=1)

        return output, hidden, attn_weights


# bahdanau 2015
class Attention(nn.Module):

    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()

        self.h_hidden_enc = n_hidden_enc
        self.h_hidden_dec = n_hidden_dec

        self.W = nn.Linear(2 * n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False)
        self.V = nn.Parameter(torch.rand(n_hidden_dec))

    def forward(self, hidden_dec, last_layer_enc):
        '''
            PARAMS:
                hidden_dec:     [b, n_layers, n_hidden_dec]    (1st hidden_dec = encoder's last_h's last layer)
                last_layer_enc: [b, seq_len, n_hidden_enc * 2]

            RETURN:
                att_weights:    [b, src_seq_len]
        '''

        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)  # [b, src_seq_len, n_hidden_dec]

        tanh_W_s_h = torch.tanh(
            self.W(torch.cat((hidden_dec, last_layer_enc), dim=2)))  # [b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)  # [b, n_hidde_dec, seq_len]

        V = self.V.repeat(batch_size, 1).unsqueeze(1)  # [b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)  # [b, seq_len]

        att_weights = F.softmax(e, dim=1)  # [b, src_seq_len]

        return att_weights
