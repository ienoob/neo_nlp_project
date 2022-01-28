#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import argparse
import torch.nn as nn
# from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from pytorch.layers.bert_optimization import BertAdam
from pytorch.entity_link.data_prepare import EntityLinkDataset, link_train_list, link_dev_list


class BertEntityModel(nn.Module):
    def __init__(self, config):
        super(BertEntityModel, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        self.classifier = nn.Linear(config.hidden_feature, 1)
        self.activate = nn.Sigmoid()

    def forward(self, token_id, seg_id, bert_mask):
        outputs = self.bert_encoder(token_id, token_type_ids=seg_id, attention_mask=bert_mask)

        outputs = outputs[0][:,0,:]

        logits = self.classifier(outputs)
        logits = self.activate(logits)

        return logits

def evaluation(model, data_iterator):
    hit = 0.0
    count = 0.0
    for idx, batch_data in enumerate(data_iterator):
        link_logtis = model(batch_data["batch_input_ids"],
                         batch_data["batch_token_type_id"],
                         batch_data["batch_attention_mask"])

        link_logtis = link_logtis.cpu().numpy()
        batch_num_value = link_logtis.shape[0]
        batch_label = batch_data["batch_input_labels"].cpu().numpy()

        for b in range(batch_num_value):
            link_pred = link_logtis[b]
            link_real = batch_label[b]

            if link_pred == link_real:
                hit += 1
            count += 1

    print("accuracy {}".format(hit/count))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--hidden_feature', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)

    config = parser.parse_args()


    loss_fn = nn.BCELoss(reduction='mean')

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    model = BertEntityModel(config)

    train_dataset = EntityLinkDataset(link_train_list, tokenizer)
    dev_dataset = EntityLinkDataset(link_dev_list, tokenizer)
    data_loader = train_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    dev_loader = dev_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

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

    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(data_loader):
            link_logtis = model(batch_data["batch_input_ids"],
                               batch_data["batch_token_type_id"],
                               batch_data["batch_attention_mask"])
            loss = loss_fn(link_logtis, batch_data["batch_input_labels"])
            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                # loss_list.append(loss_value)

            # print(loss.data)

            loss.backward()
            optimizer.step()
            model.zero_grad()
        evaluation(model, dev_loader)


if __name__ == "__main__":
    train()
