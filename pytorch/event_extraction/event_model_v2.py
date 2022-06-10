#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertTokenizerFast

from pytorch.layers.bert_optimization import BertAdam
from pytorch.syntactic_parsing.parser_layer import NonLinear, Biaffine
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3


class EventDataset(Dataset):

    def __init__(self, document_list, tokenizer, label2id, id2label):
        super(EventDataset, self).__init__()
        self.document_list = document_list
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label

    def __getitem__(self, item):
        """
            Args:
                item: int, idx
        """
        document = self.document_list[item]
        return document

    def __len__(self):
        return len(self.document_list)

    def _create_collate_fn(self, batch_first=False):
        def collate(documents):
            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_input_labels = []
            batch_gold_answer = []

            local_max = 0
            for document in documents:
                text = document["text"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                gold_answer = []
                text = document["text"]
                text_word = [t for t in list(text)]
                codes = self.tokenizer.encode_plus(text_word,
                                                   return_offsets_mapping=True,
                                                   is_split_into_words=True,
                                                   max_length=local_max,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")

                input_ids_ = codes["input_ids"]
                label_metrix = torch.zeros((local_max, local_max))
                # print(text_word)

                # print(codes["offset_mapping"])
                iiv = len(input_ids_) - 1
                while input_ids_[iiv] == 0:
                    iiv -= 1
                # print(iiv + 1, len(document["label"]), len(text_word), len(input_ids_), local_max)
                # print(input_ids_)
                assert iiv + 1 == len(document["label"]) + 2

                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)

                label_seq = [self.id2label[la] for la in document["label"]]
                entities = extract_entity(label_seq)
                for entity_s, entity_e, entity_id in entities:
                    label_metrix[entity_s+1][entity_e] = int(entity_id)+1
                    gold_answer.append((entity_s+1, entity_e, int(entity_id)+1))
                batch_input_labels.append(label_metrix)
                batch_gold_answer.append(gold_answer)
            batch_input_labels = torch.stack(batch_input_labels, dim=0).long()
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_input_labels": batch_input_labels,
                "batch_gold_answer": batch_gold_answer
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class RobertaBiaffineNER(nn.Module):
    def __init__(self, config):
        super(RobertaBiaffineNER, self).__init__()

        # self.gpu = gpu_use
        # self.data_bertpath = bertpath
        # self.bertpath = self.data_bertpath

        # char_feature_dim = 768
        # print('total char_feature_dim is {}'.format(config.char_feature_dim))

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)

        self.mlp_arc_dep = NonLinear(
            input_size=config.bert_hidden,
            hidden_size=config.mlp_arc_hidden,
            activation=nn.LeakyReLU(0.1))

        self.mlp_arc_head = NonLinear(
            input_size=config.bert_hidden,
            hidden_size=config.mlp_arc_hidden,
            activation=nn.LeakyReLU(0.1))

        self.ner_biaffine = Biaffine(config.mlp_arc_hidden, config.mlp_arc_hidden, config.entity_size, bias=(True, True))

    def forward(self, token_id, seg_id, bert_mask):
        outputs = self.bert_encoder(token_id, token_type_ids=seg_id, attention_mask=bert_mask)
        outputs = outputs[0]

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        ner_logit_cond = self.ner_biaffine(x_all_dep, x_all_head)

        return ner_logit_cond


class RobertaSpanNer(nn.Module):
    def __init__(self, config):
        super(RobertaSpanNer, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)

        self.activation = nn.ReLU()

        self.rel_emb = nn.Embedding(num_embeddings=config.entity_num, embedding_dim=config.entity_emb_size)

        self.selection_u = nn.Linear(config.hidden_size, config.entity_num)
        self.selection_v = nn.Linear(config.hidden_size, config.entity_num)
        self.selection_uv = nn.Linear(config.entity_num*2, config.entity_num)

    def forward(self, token_id, seg_id, bert_mask):
        outputs = self.bert_encoder(token_id, token_type_ids=seg_id, attention_mask=bert_mask)
        outputs = outputs[0]

        B, L, H = outputs.size()
        u = self.activation(self.selection_u(outputs)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(outputs)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        entity_logits = torch.einsum('bijh,rh->birj', uv, self.entity_emb.weight)

        return entity_logits


def evaluation(model, data_iterator):
    model.eval()
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0

    role_indicate = dict()

    for idx, batch_data in enumerate(data_iterator):
        tag_seqs = model(batch_data["batch_input_ids"],
                         batch_data["batch_token_type_id"],
                         batch_data["batch_attention_mask"])

        # tag_seqs = tag_seqs * batch_data["batch_attention_mask"]
        tag_seqs = tag_seqs.detach().numpy()

        # print(tag_seqs.shape)
        # label_seq = batch_data["batch_input_labels"].numpy()
        batch_num= tag_seqs.shape[0]

        for b in range(batch_num):
            gold_ans = batch_data["batch_gold_answer"][b]
            tag_pred_list = tag_seqs[b]
            # print(tag_pred_list.shape)
            pre_value = []
            for ii, row in enumerate(tag_pred_list):
                for jj, rx in enumerate(row):
                    pred_entity = np.argmax(rx)
                    if pred_entity == 0:
                        continue
                    pre_value.append((ii, jj, pred_entity))
                    if (ii, jj, pred_entity) in gold_ans:
                        hit_num += 1

            pre_num += len(pre_value)
            true_num += len(gold_ans)


            # tag_seq_list = [id2label.get(tag, "O") for tag in tag_seq_list]
            # # print(tag_seq_list)
            #
            # true_seq_list = label_seq[b]
            # true_seq_list = [id2label.get(tag, "O") for tag in true_seq_list]
            #
            # pre_value = extract_entity(tag_seq_list)
            # true_value = extract_entity(true_seq_list)
            #
            # for e in true_value:
            #     role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
            #     role_indicate[e[2]]["real"] += 1
            #
            # for e in pre_value:
            #     role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
            #     role_indicate[e[2]]["pred"] += 1
            # # print(true_value)
            #
            # pre_num += len(pre_value)
            # true_num += len(true_value)
            #
            # for p in pre_value:
            #     if p in true_value:
            #         hit_num += 1
            #         role_indicate[p[2]]["hit"] += 1
        print(hit_num, true_num, pre_num)
    # for role_id, role_ind in role_indicate.items():
    #     print("{} : {}".format(id2role[int(role_id)],
    #                            eval_metrix_v3(role_ind["hit"], role_ind["real"], role_ind["pred"])))
    metric = eval_metrix_v3(hit_num, true_num, pre_num)
    return {
        "hit_num": hit_num,
        "true_num": true_num,
        "pre_num": pre_num,
        "recall": metric["recall"],
        "precision": metric["precision"],
        "f1_value": metric["f1_value"]
    }


def train(rt_data):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--bert_hidden', type=int, default=768, required=False)
    parser.add_argument('--mlp_arc_hidden', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_size', type=int, default=0, required=False)

    config = parser.parse_args()

    def compute_loss(entity_logits, entity_true):
        # print(entity_logits.shape)
        b, l1, l1, l2 = entity_logits.size()
        arc_loss = F.cross_entropy(
            entity_logits.view(b * l1*l1, l2), entity_true.view(b * l1*l1),
            ignore_index=-1)

        return arc_loss

    tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # train_data_list, label2id = train_dataset()
    train_data_list = rt_data["bert_data"]["all"]

    role2id = rt_data["role2id_v2"]
    label2id = rt_data["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    config.entity_size = len(role2id)
    train_dataset = EventDataset(train_data_list, tokenizer, role2id, id2label)

    data_loader = train_dataset.get_dataloader(config.batch_size,
                                                 shuffle=config.shuffle,
                                                 pin_memory=config.pin_memory)

    model = RobertaBiaffineNER(config)

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,
                         t_total=int(len(train_dataset) // config.batch_size + 1) * config.epoch)
    loss_list = []

    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(data_loader):
            tag_logits = model(batch_data["batch_input_ids"],
                                  batch_data["batch_token_type_id"],
                                  batch_data["batch_attention_mask"])
            loss = compute_loss(tag_logits, batch_data["batch_input_labels"])

            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                loss_list.append(loss_value)

            # print(loss.data)

            loss.backward()
            optimizer.step()
            model.zero_grad()
        eval_res = evaluation(model, data_loader)
        print(eval_res)
    plt.plot(loss_list)
    plt.show()
    #
        # break

