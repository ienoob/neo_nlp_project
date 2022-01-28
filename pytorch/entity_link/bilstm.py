#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import argparse
import torch
import torch.nn as nn
from  torch import optim


class LinkBiLstmModel(nn.Module):

    def __init__(self, config):
        super(LinkBiLstmModel, self).__init__()

        self.embed = nn.Embedding(config.word_size, config.embed_size)
        # 官方的实现有问题就是non-reproducible. 也就是这个双向LSTM, 每次出现的结果会有不同。加上dropout 就会出现这个问题
        self.bilstm_mention = nn.LSTM(input_size=config.embed_size, hidden_size=config.lstm_size,
                              num_layers=config.num_layers, batch_first=True,
                              dropout=config.dropout,
                              bidirectional=True)

        self.bilstm_entity = nn.LSTM(input_size=config.embed_size, hidden_size=config.lstm_size,
                              num_layers=config.num_layers, batch_first=True,
                              dropout=config.dropout,
                              bidirectional=True)

        self.out = nn.Linear(config.lstm_size*4, 1)
        # self.big_w = nn.Linear(config.lstm_size * 4, 100)
        # self.attn = nn.Linear(config.lstm_size*2, config.lstm_size*2)
        self.activate = nn.Sigmoid()

    def forward(self, mention_id, entity_id):

        mention_embed = self.embed(mention_id)
        entity_embed = self.embed(entity_id)

        mention_lstm, _ = self.bilstm_mention(mention_embed)
        entity_lstm, _ = self.bilstm_entity(entity_embed)

        mention_item = mention_lstm[:, 0, :]
        entity_item = entity_lstm[:, 0, :]

        # (1)
        mention_entity = torch.cat((mention_item, entity_item), dim=-1)

        logit = self.out(mention_entity)
        logit = self.activate(logit)
        #

        # (2) cosine
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # logit = cos(mention_item, entity_item)

        # (3 )additive
        # mention_entity = torch.cat((mention_item, entity_item), dim=-1)
        # out = self.big_w(mention_entity)
        # logit = nn.Tanh()(out)
        #
        # v = nn.Parameter(torch.rand(100))
        # # print(v.repeat(2, 1).shape)
        # v = v.repeat(2, 1).unsqueeze(1)
        # logit = logit.unsqueeze(1).permute(0, 2, 1)
        #
        # logit = torch.bmm(v, logit).squeeze(1)

        # (4)
        # attn_value = self.attn(mention_item).unsqueeze(1)
        # entity_item = entity_item.unsqueeze(1).permute(0, 2, 1)
        # logit = torch.bmm(attn_value, entity_item).squeeze(1)
        #
        # logit = self.activate(logit)
        return logit

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--word_size', type=int, default=100, required=False)
    parser.add_argument('--embed_size', type=int, default=128, required=False)
    parser.add_argument("--lstm_size", type=int, default=128, required=False)
    parser.add_argument("--num_layers", type=int, default=2, required=False)
    parser.add_argument("--dropout", type=float, default=0.5, required=False)

    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)

    config = parser.parse_args()

    model = LinkBiLstmModel(config)

    test_mention = torch.tensor([[1, 2, 3], [2, 3, 4]])
    test_entity = torch.tensor([[1, 2, 3, 4], [3, 2, 4, 1]])

    model.eval()

    res = model(test_mention, test_entity)

    print(res)

def evaluation(model, data_iterator, id2label, id2role):
    pass


def train():
    from pytorch.entity_link.data_prepare import char2id, datasetv1

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--word_size', type=int, default=len(char2id), required=False)
    parser.add_argument('--embed_size', type=int, default=128, required=False)
    parser.add_argument("--lstm_size", type=int, default=128, required=False)
    parser.add_argument("--num_layers", type=int, default=2, required=False)
    parser.add_argument("--dropout", type=float, default=0.5, required=False)

    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)

    config = parser.parse_args()

    model = LinkBiLstmModel(config)
    data_loader = datasetv1.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    loss_fn = nn.BCELoss(reduce=True, size_average=False)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters)

    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(data_loader):
            link_logtis = model(batch_data["batch_mention_ids"],
                               batch_data["batch_entity_ids"])
            loss = loss_fn(link_logtis, batch_data["batch_input_labels"])
            if idx % 100 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))

            loss.backward()
            optimizer.step()
            model.zero_grad()
        break


train()
