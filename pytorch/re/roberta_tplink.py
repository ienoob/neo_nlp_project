#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoModel, BertTokenizerFast
from pytorch.re.tplink_functions import HandshakingKernel
from torch.autograd import Variable


class RobertaTplink(nn.Module):

    def __init__(self, config):
        super(RobertaTplink, self).__init__()

        self.roberta_embed = BertModel.from_pretrained(config.pretrain_name)
        # self.rel_emb = nn.Embedding(num_embeddings=config.rel_num, embedding_dim=config.rel_emb_size)
        self.activation = nn.ReLU()

        self.eh_eh_u = nn.Linear(config.hidden_size, config.eh_size)
        self.eh_eh_v = nn.Linear(config.hidden_size, config.eh_size)
        self.eh_eh_uv = nn.Linear(config.eh_size*2, 1)

        self.h_h_u = nn.Linear(config.hidden_size, config.rel_emb_size)
        self.h_h_v = nn.Linear(config.hidden_size, config.rel_emb_size)
        self.h_h_uv = nn.Linear(config.rel_emb_size*2, config.rel_emb_size)
        self.h_h_classifier = nn.Linear(config.rel_emb_size, config.rel_num)

        self.t_t_u = nn.Linear(config.hidden_size, config.rel_emb_size)
        self.t_t_v = nn.Linear(config.hidden_size, config.rel_emb_size)
        self.t_t_uv = nn.Linear(config.rel_emb_size*2, config.rel_emb_size)
        self.t_t_classifier = nn.Linear(config.rel_emb_size, config.rel_num)

    def forward(self, input_ids, input_masks=None):
        if input_masks is None:
            input_masks = input_ids.gt(0)
        encoder = self.roberta_embed(input_ids, input_masks)
        encoder = encoder[0]

        B, L, H = encoder.size()
        # print(B, L, H)
        eh_u = self.activation(self.eh_eh_u(encoder)).unsqueeze(1).expand(B, L, L, -1)
        eh_v = self.activation(self.eh_eh_v(encoder)).unsqueeze(2).expand(B, L, L, -1)
        eh_uv = self.activation(self.eh_eh_uv(torch.cat((eh_u, eh_v), dim=-1)))

        eh_logits = eh_uv

        hh_u = self.activation(self.h_h_u(encoder)).unsqueeze(1).expand(B, L, L, -1)
        hh_v = self.activation(self.h_h_v(encoder)).unsqueeze(2).expand(B, L, L, -1)
        hh_uv = self.activation(self.h_h_uv(torch.cat((hh_u, hh_v), dim=-1)))

        # hh_selection_logits = torch.einsum('bijh,rh->birj', hh_uv, self.rel_emb.weight)
        hh_selection_logits = self.h_h_classifier(hh_uv)
        hh_logits = nn.Softmax(dim=-1)(hh_selection_logits)
        # hh_logits = hh_logits.permute(0, 1, 3, 2)

        tt_u = self.activation(self.t_t_u(encoder)).unsqueeze(1).expand(B, L, L, -1)
        tt_v = self.activation(self.t_t_v(encoder)).unsqueeze(2).expand(B, L, L, -1)
        tt_uv = self.activation(self.t_t_uv(torch.cat((tt_u, tt_v), dim=-1)))

        # tt_selection_logits = torch.einsum('bijh,rh->birj', tt_uv, self.rel_emb.weight)
        tt_selection_logits = self.t_t_classifier(tt_uv)
        tt_logits = nn.Softmax(dim=-1)(tt_selection_logits)
        # tt_logits = tt_logits.permute(0, 1, 3, 2)

        return eh_logits, hh_logits, tt_logits


class BertTplinkV2(nn.Module):

    def __init__(self, config):
        super(BertTplinkV2, self).__init__()
        self.bert_embed = BertModel.from_pretrained(config.pretrain_name)

        self.ent_fc = nn.Linear(config.hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(config.hidden_size, 3) for _ in range(config.rel_size)]
        self.tail_rel_fc_list = [nn.Linear(config.hidden_size, 3) for _ in range(config.rel_size)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        self.handshaking_kernel = HandshakingKernel(config.hidden_size, config.shaking_type, config.inner_enc_type)

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.bert_embed(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]

        # print(last_hidden_state.shape)

        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        # print(shaking_hiddens.shape)
        # shaking_hiddens4ent = shaking_hiddens
        # shaking_hiddens4rel = shaking_hiddens

        ent_shaking_outputs = self.ent_fc(shaking_hiddens)
        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens))

        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


import argparse

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--hidden_size", type=int, default=768, required=False)
    parser.add_argument("--rel_size", type=int, default=3, required=False)
    parser.add_argument("--shaking_type", type=str, default="cat", required=False)
    parser.add_argument("--inner_enc_type", type=str, default="lstm", required=False)

    config = parser.parse_args()

    model = BertTplinkV2(config)

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name, add_special_tokens=False, do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: \
    tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
    text = "当你的才华还撑不起你的梦想时"
    codes = tokenizer.encode_plus(text,
                                       return_offsets_mapping=True,
                                       add_special_tokens=False,
                                       max_length=512,
                                       truncation=True,
                                       pad_to_max_length=True)

    print(codes)



    tokens = tokenize(text)
    tok2char_span = get_tok2char_span_map(text)
    # token_id = tokenizer.encode_plus
    print(tokens)
    print(tok2char_span[0])

    print(tokens[0])

    print(tokenizer.encode(text))

    token_ids = tokenizer.encode(text, return_tensors="pt")
    print(token_ids)
    token_mask = torch.tensor([[1]*len(token_ids)])
    seg_ids = torch.tensor([[1] * len(token_ids)])

    model(Variable(token_ids).to(device),
          Variable(token_mask).to(device),
          Variable(seg_ids).to(device))



