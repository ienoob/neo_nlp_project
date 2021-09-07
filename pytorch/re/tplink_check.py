#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import argparse
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
from typing import List
from functools import partial
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from nlp_applications.ie_relation_extraction.evaluation import eval_metrix
from pytorch.layers.bert_optimization import BertAdam
from torch.utils.data import Dataset, DataLoader
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from pytorch.re.roberta_tplink import BertTplinkV2
from pytorch.re.duie2_bert_tplink import Duie2Dataset, data_loader, shaking_ind2matrix_ind, matrix_ind2shaking_ind, max_len

device = "cpu"
if torch.cuda.is_available():
    device = "gpu"

print(device)
def check_fun():
    data_path = "D:\data\关系抽取"
    # data_loader = LoaderDuie2Dataset(data_path)

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--hidden_size", type=int, default=768, required=False)
    parser.add_argument("--rel_size", type=int, default=len(data_loader.relation2id), required=False)
    parser.add_argument("--shaking_type", type=str, default="cat", required=False)
    parser.add_argument("--inner_enc_type", type=str, default="lstm", required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)

    config = parser.parse_args()

    dataset = Duie2Dataset(data_loader.documents, config.pretrain_name, config.rel_size)

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                                   shuffle=config.shuffle,
                                                   pin_memory=config.pin_memory)

    for batch in train_data_loader:
        batch_ent2ent_seq_tag = batch["batch_ent2ent_seq_tag"]
        batch_head2head_seq_tag = batch["batch_head2head_seq_tag"]
        batch_tail2tail_seq_tag = batch["batch_tail2tail_seq_tag"]
        batch_gold_answer = batch["batch_gold_answer"]
        batch_text = batch["batch_text"]
        batch_token_offset = batch["batch_token_offset"]
        batch_ent2ent_seq_mask = batch["batch_ent2ent_seq_mask"]


        batch_size = batch_ent2ent_seq_tag.shape[0]

        for b in range(batch_size):
            text = batch_text[b]
            token_offset = batch_token_offset[b]
            ent2ent_seq_tag = batch_ent2ent_seq_tag[b]
            head2head_seq_tag = batch_head2head_seq_tag[b]
            tail2tail_seq_tag = batch_tail2tail_seq_tag[b]
            gold_answer = batch_gold_answer[b]

            ent2ent_seq_tag = ent2ent_seq_tag*batch_ent2ent_seq_mask[b]

            real_entity = set()
            real_h2h = set()
            real_t2t = set()
            for ga in gold_answer:
                real_entity.add((ga[0], ga[1]))
                real_entity.add((ga[2], ga[3]))
                real_h2h.add((ga[0], ga[2], ga[4]))
                real_t2t.add((ga[1], ga[3], ga[4]))

            entity_list = set()
            for xi in ent2ent_seq_tag.nonzero(as_tuple=False):
                xv = xi[0]
                start, end = shaking_ind2matrix_ind[xv]
                entity_list.add((start, end))
            # print(entity_list)
            assert len(entity_list) == len(real_entity)

            h2h_list = []
            for xi in head2head_seq_tag.nonzero(as_tuple=False):
                xr = xi[0].numpy()
                xv = xi[1].numpy()
                x_tag = head2head_seq_tag[xr][xv].numpy()
                if x_tag == 1:
                    s_start, o_start = shaking_ind2matrix_ind[xv]
                    h2h_list.append((s_start, o_start, xr))
                elif x_tag == 2:
                    o_start, s_start = shaking_ind2matrix_ind[xv]
                    h2h_list.append((s_start, o_start, xr))

            # print(real_h2h)
            # print(h2h_list)

            t2t_list = []
            for xi in tail2tail_seq_tag.nonzero(as_tuple=False):
                xr = xi[0].numpy()
                xv = xi[1].numpy()
                x_tag = tail2tail_seq_tag[xr][xv].numpy()
                if x_tag == 1:
                    s_end, o_end = shaking_ind2matrix_ind[xv]
                    t2t_list.append((s_end, o_end, xr))
                elif x_tag == 2:
                    o_end, s_end = shaking_ind2matrix_ind[xv]
                    t2t_list.append((s_end, o_end, xr))

            print(real_t2t)
            print(t2t_list)
            predict_list = []
            predict_list_text = []
            for ss, os, sr in h2h_list:
                for st, ot, tr in t2t_list:
                    if (ss, st) not in entity_list:
                        continue
                    if (os, ot) not in entity_list:
                        continue
                    if sr != tr:
                        continue
                    predict_list.append((ss, st, os, ot, sr))
                    real_ss = token_offset[ss][0]
                    real_st = token_offset[st][1]
                    real_os = token_offset[os][0]
                    real_ot = token_offset[ot][1]
                    predict_list_text.append((text[real_ss:real_st], text[real_os:real_ot]))
            print(text)
            print(gold_answer)
            print(predict_list)
            print(predict_list_text)







        break

def check_fun2():
    data_path = "D:\data\关系抽取"
    # data_loader = LoaderDuie2Dataset(data_path)

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--hidden_size", type=int, default=768, required=False)
    parser.add_argument("--rel_size", type=int, default=len(data_loader.relation2id), required=False)
    parser.add_argument("--shaking_type", type=str, default="cat", required=False)
    parser.add_argument("--inner_enc_type", type=str, default="lstm", required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--shuffle", type=bool, default=False, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    # print("hello v1")

    config = parser.parse_args()

    dataset = Duie2Dataset(data_loader.documents, config.pretrain_name, config.rel_size)
    # print("hello v3")

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                                   shuffle=config.shuffle,
                                                   pin_memory=config.pin_memory)
    # print("hello v4")
    epoch = 0
    loss = 0
    for step, batch in enumerate(train_data_loader):
        batch_ent2ent_seq_tag = batch["batch_ent2ent_seq_tag"]
        batch_head2head_seq_tag = batch["batch_head2head_seq_tag"]
        batch_tail2tail_seq_tag = batch["batch_tail2tail_seq_tag"]
        batch_gold_answer = batch["batch_gold_answer"]
        # batch_text = batch["batch_text"]
        # batch_token_offset = batch["batch_token_offset"]
        batch_ent2ent_seq_mask = batch["batch_ent2ent_seq_mask"]


        batch_size = batch_ent2ent_seq_tag.shape[0]

        print(
                    u"step {} / {} of epoch {}, train/loss: {}".format(step, len(train_data_loader),
                                                                       epoch, loss))

check_fun2()

# for i in range(10):
#     start = i*(max_len-i)+i+i
#     end = i*(max_len-i)+i+10
#     print(list(range(start, end)))
#     print("+++++++++++")
#     for j in range(i, 10):
#         iv = matrix_ind2shaking_ind[i][j]
#         print(iv)
