#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from functools import partial
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from pytorch.layers.bert_optimization import BertAdam

#
# class TextMatchDataset(Dataset):
#
#     def __init__(self):
#         super(TextMatchDataset, self).__init__()

def sequence_padding(inputs, length=None, padding=0, is_float=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return out_tensor.clone().detach()

class BertTextMatch(nn.Module):

    def __init__(self, config):
        super(BertTextMatch, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        self.classifier = nn.Linear(config.feature_num, 1)
        self.activate = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        bert_pool = bert_outputs[0][:,0]
        is_match = self.classifier(bert_pool)
        is_match = self.activate(is_match)
        return is_match

data_path = "D:\\data\\文本匹配\\paws-x-zh\\train.tsv"
dev_path = "D:\\data\\文本匹配\\paws-x-zh\\dev.tsv"

with open(data_path, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = f.read()

train_data_list = train_data.split("\n")
dev_data_list = dev_data.split("\n")
train_data_ = []
for data_item in train_data_list:
    data_item = data_item.strip()
    if len(data_item) == 0:
        continue
    if len(data_item.split("\t")) == 1:
        # print(data, "hello")
        continue
    sentence1, sentence2, label = data_item.split("\t")
    train_data_.append((sentence1, sentence2, int(label)))

dev_data_ = []
for data_item in dev_data_list:
    data_item = data_item.strip()
    if len(data_item) == 0:
        continue
    if len(data_item.split("\t")) == 1:
        # print(data, "hello")
        continue
    sentence1, sentence2, label = data_item.split("\t")
    dev_data_.append((sentence1, sentence2, int(label)))

bert_model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)


class Paws2Dataset(Dataset):

    def __init__(self, data_list):
        super(Paws2Dataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def _create_collate_fn(self, batch_first=False):

        def collate(documents):

            batch_input_ids = []
            batch_token_type_ids = []
            batch_attention_mask = []
            batch_match_label = []

            for document in documents:

                text1 = document[0]
                text2 = document[1]
                text_label = document[2]

                query_context_tokens = tokenizer.encode_plus(text1, text2, add_special_tokens=True,
                                                             return_offsets_mapping=True)
                tokens = query_context_tokens["input_ids"]
                type_ids = query_context_tokens["token_type_ids"]
                attention_mask = query_context_tokens["attention_mask"]

                batch_input_ids.append(tokens)
                batch_token_type_ids.append(type_ids)
                batch_attention_mask.append(attention_mask)
                batch_match_label.append([text_label])

            batch_input_ids = sequence_padding(batch_input_ids)
            batch_token_type_ids = sequence_padding(batch_token_type_ids)
            batch_attention_mask = sequence_padding(batch_attention_mask)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_token_type_ids": batch_token_type_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_match_label": torch.FloatTensor(batch_match_label),
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)



def evaluation(model, dataloader):
    model.eval()
    acc_num = 0.0
    for step, batch in enumerate(dataloader):
        predict_res = model(batch["batch_input_ids"],
                            batch["batch_token_type_ids"],
                            batch["batch_attention_mask"])
        batch_label = batch["batch_match_label"]
        for i, p in enumerate(predict_res):
            p_label = 1 if p>0.5 else 0
            if p_label == batch_label[i]:
                acc_num += 1

    return acc_num/len(dataloader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default=bert_model_name, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--feature_num", type=int, default=768, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--mrc_dropout', type=float, default=0.5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    config = parser.parse_args()

    dataset = Paws2Dataset(train_data_)
    dev_dataset = Paws2Dataset(dev_data_)
    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    dev_data_loader = dev_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    model = BertTextMatch(config)
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = optim.Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,
                         t_total=int(len(dataset) // config.batch_size + 1) * config.epoch)

    loss_func = nn.BCELoss()

    for epoch in range(config.epoch):
        start = time.time()
        for step, batch in enumerate(train_data_loader):

            model.train()
            predict_res = model(batch["batch_input_ids"],
                                batch["batch_token_type_ids"],
                                batch["batch_attention_mask"])
            loss = loss_func(predict_res, batch["batch_match_label"])

            if step % 100 == 0:
                cost_time = time.time() - start
                print(u"step {0} / {1} of epoch {2}, train/loss: {3}, cost {4}".format(step, len(train_data_loader),
                                                                       epoch, loss, cost_time))
                start = time.time()


            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        acc = evaluation(model, dev_data_loader)
        print("epoch {} accuracy {}".format(epoch, acc))
        # break





