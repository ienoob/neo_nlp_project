#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/21 15:09
    @Author  : jack.li
    @Site    : 
    @File    : bert_mrc_model+v2.py

"""
import re
import torch
import argparse

import pandas as pd
from torch import nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertModel, BertTokenizer, BertTokenizerFast
# from nlp_applications.data_loader import LoaderDuReaderChecklist

path = "D:\\data\\百度比赛\\2021语言与智能技术竞赛：机器阅读理解任务\\dureader_checklist.dataset\dataset\\train.json"
# data_loader = LoaderDuReaderChecklist(path)

def get_train_data():
    # with open(path, "r") as f:
    #     data = f.read()
    xpath = "D:\\xxxx\\brand_1000.csv"
    brand_1000 = pd.read_csv(xpath)

    train_list = []
    for idx, row in brand_1000.iterrows():
        brand = row["brand"]
        sentence = row["sentence"]
        ans = []
        if brand and not isinstance(brand, float):
            for sub_brand in brand.split("、"):
                for res in re.finditer(sub_brand, sentence):
                    # print(res.span())
                    ans.append(res.span())
        train_list.append({"question": "有什么品牌？", "sentence": row["sentence"], "ans": ans})
    return train_list


class DuReaderDataset(Dataset):
    def __init__(self, qa_list, tokenizer):
        super(DuReaderDataset, self).__init__()
        self.qa_list = qa_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        """
            Args:
                item: int, idx
        """
        document = self.qa_list[item]
        return document

    def __len__(self):
        return len(self.qa_list)

    def _create_collate_fn(self, batch_first=False):
        def collate(documents):

            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_start_labels = []
            batch_end_labels = []

            local_max = 0
            for document in documents:
                text = document["question"] + document["sentence"]
                local_max = max(len(text), local_max)
            local_max += 3
            for document in documents:

                codes = self.tokenizer.encode_plus(document["question"],
                                                   document["sentence"],
                                                   return_offsets_mapping=True,
                                                   max_length=local_max,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")
                input_ids_ = codes["input_ids"]

                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_start_label = torch.zeros(local_max)
                batch_end_label = torch.zeros(local_max)
                for start, end in document["ans"]:
                    batch_start_label[start+len(document["question"])+2] = 1
                    batch_end_label[end+len(document["question"])+1] = 1

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                batch_start_labels.append(batch_start_label)
                batch_end_labels.append(batch_end_label)
            #     batch_input_labels.append(torch.tensor([document["link"]]))
            # batch_input_labels = torch.stack(batch_input_labels, dim=0).float()
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
            batch_start_labels = torch.stack(batch_start_labels, dim=0)
            batch_end_labels = torch.stack(batch_end_labels, dim=0)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_start_labels": batch_start_labels,
                "batch_end_labels": batch_end_labels
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class MrcModel(nn.Module):

    def __init__(self, config):
        super(MrcModel, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        self.start_indx = nn.Linear(config.hidden_size, 1)
        self.end_indx = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        # print(sequence_heatmap.shape)
        # batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_indx(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_indx(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        return start_logits, end_logits



def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--mrc_dropout', type=float, default=0.5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    train_list = get_train_data()
    dataset = DuReaderDataset(train_list, tokenizer)
    dataloader = dataset.get_dataloader(config.batch_size,
                                        shuffle=config.shuffle,
                                        pin_memory=config.pin_memory)

    model = MrcModel(config)
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
                         t_total=int(len(train_list) // config.batch_size + 1) * config.epoch)
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    for epoch in range(config.epoch):
        for idx, batch_data in enumerate(dataloader):
            # print(data["batch_start_labels"].shape)
            start_logits, end_logits = model(batch_data["batch_input_ids"],
                                batch_data["batch_token_type_id"],
                                batch_data["batch_attention_mask"])
            # print(start_logits.shape)

            start_loss = bce_loss(start_logits.view(-1), batch_data["batch_start_labels"].view(-1).float())
            end_loss = bce_loss(end_logits.view(-1), batch_data["batch_end_labels"].view(-1).float())
            loss = start_loss + end_loss

            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                # loss_list.append(loss_value)

            # print(loss.data)

            loss.backward()
            optimizer.step()
            model.zero_grad()
        predict(model, "有什么品牌？", "销售业绩大跌 运动品牌受疫情冲击叫苦不迭")
        # break

def predict(model, question, sentence):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
    batch_start_labels = []
    batch_end_labels = []

    local_max = len(question) + len(sentence) + 3

    codes = tokenizer.encode_plus(question,
                                       sentence,
                                       return_offsets_mapping=True,
                                       max_length=local_max,
                                       truncation=True,
                                       return_length=True,
                                       padding="max_length")
    input_ids_ = codes["input_ids"]

    input_ids = torch.tensor(input_ids_).long()
    attention_mask = torch.tensor(codes["attention_mask"]).long()
    token_type_ids = torch.tensor(codes["token_type_ids"]).long()

    batch_input_ids.append(input_ids)
    batch_attention_mask.append(attention_mask)
    batch_token_type_id.append(token_type_ids)

    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
    batch_token_type_id = torch.stack(batch_token_type_id, dim=0)

    start_logits, end_logits = model(batch_input_ids,
                                     batch_token_type_id,
                                     batch_attention_mask)

    start = torch.where(start_logits > 0.5)[0]
    end = torch.where(end_logits > 0.5)[0]
    res = []
    for i in start:
        j = end[end >= i]
        if i == 0 or i > local_max:
            continue

        if len(j) > 0:
            j = j[0]
            if j > local_max - 3:
                continue
            # res.append((i, j))
            print("ans", sentence[i-len(question)-1:j-len(question)])



    # print(start)
    # print(end)




class Extractor(object):

    def __init__(self, model_path):
        self.model = None





if __name__ == "__main__":
    train()
