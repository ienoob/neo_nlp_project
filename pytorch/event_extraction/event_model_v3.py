#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch.utils.data import Dataset, DataLoader
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from nlp_applications.ner.evaluation import extract_entity
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
                label_metrix = torch.zeros((local_max, len(self.label2id)*2))
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
                    label_metrix[entity_s+1][int(entity_id)*2] = 1
                    label_metrix[entity_e][int(entity_id) * 2+1] = 1
                    gold_answer.append((entity_s + 1, entity_e, int(entity_id) + 1))
                batch_input_labels.append(label_metrix)
                batch_gold_answer.append(gold_answer)
            batch_input_labels = torch.stack(batch_input_labels, dim=0).float()
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


class RobertaNetPointerNER(nn.Module):
    def __init__(self, config):
        super(RobertaNetPointerNER, self).__init__()
        self.class_num = config.entity_size

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)

        self.role_pn = nn.Linear(config.hidden_size, self.class_num*2)
        self.activate = nn.Sigmoid()
        # self.trigger = nn.Linear(config.hidden_size, 2)

    def forward(self, token_id, seg_id, bert_mask):
        outputs = self.bert_encoder(token_id, token_type_ids=seg_id, attention_mask=bert_mask)
        outputs = outputs[0]

        role_pred = self.role_pn(outputs).reshape(token_id.size(0), -1, self.class_num*2)
        role_pred = self.activate(role_pred)

        return role_pred


def evaluation(model, data_loader):
    model.eval()
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0

    role_indicate = dict()

    for idx, batch_data in enumerate(data_loader):
        tag_seqs = model(batch_data["batch_input_ids"],
                         batch_data["batch_token_type_id"],
                         batch_data["batch_attention_mask"])

        # tag_seqs = tag_seqs * batch_data["batch_attention_mask"]
        tag_seqs = tag_seqs.detach().numpy()

        batch_num = tag_seqs.shape[0]

        for b in range(batch_num):
            gold_ans = batch_data["batch_gold_answer"][b]
            tag_pred_list = tag_seqs[b]
            # print(tag_pred_list.shape)
            pre_value_d = dict()
            for ii, row in enumerate(tag_pred_list):
                for jj, rx in enumerate(row):
                    if rx > 0.5:
                        entity_id = int(jj//2)
                        pre_value_d.setdefault(entity_id, {"start": [], "end": []})
                        if entity_id%2==0:
                            pre_value_d[entity_id]["start"].append(rx)
                        else:
                            pre_value_d[entity_id]["end"].append(rx)
            pre_value = []
            for entity_id, info in pre_value_d.items():
                ii = 0
                jj = 0
                while jj < len(info["end"]) and ii < len(info["start"]):
                    if info["start"][ii] <= info["end"][jj]:
                        entity_p = (info["start"][ii], info["end"][jj], entity_id)
                        pre_value.append(entity_p)
                        if entity_p in gold_ans:
                            hit_num += 1
                        ii += 1
                        jj += 1
                    else:
                        jj += 1

            pre_num += len(pre_value)
            true_num += len(gold_ans)
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
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_size', type=int, default=0, required=False)

    config = parser.parse_args()

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

    model = RobertaNetPointerNER(config)

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
    loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)


    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(data_loader):
            tag_logits = model(batch_data["batch_input_ids"],
                                  batch_data["batch_token_type_id"],
                                  batch_data["batch_attention_mask"])
            loss = loss_fn(tag_logits, batch_data["batch_input_labels"])

            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                loss_list.append(loss_value)

            loss.backward()
            optimizer.step()
            model.zero_grad()
        eval_res = evaluation(model, data_loader)
        print(eval_res)
    #
    #
    #
    #         # print(loss.data)
    #
    #
    #     eval_res = evaluation(model, data_loader)
    #     print(eval_res)
    # plt.plot(loss_list)
    # plt.show()





