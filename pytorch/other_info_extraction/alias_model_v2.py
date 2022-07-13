#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    简称模型v2, 使用tplink 思路来做
"""
import argparse
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from pytorch.layers.bert_optimization import BertAdam
from pytorch.other_info_extraction.get_alias_data import train_list
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from pytorch.syntactic_parsing.parser_layer import NonLinear, Biaffine


class AliasModelV2(nn.Module):
    def __init__(self, config):
        super(AliasModelV2, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)

        self.mlp_arc_start = NonLinear(
            input_size=config.hidden_size,
            hidden_size=config.mlp_arc_hidden,
            activation=nn.LeakyReLU(0.1))

        self.mlp_arc_end = NonLinear(
            input_size=config.hidden_size,
            hidden_size=config.mlp_arc_hidden,
            activation=nn.LeakyReLU(0.1))

        self.ner_biaffine = Biaffine(config.mlp_arc_hidden, config.mlp_arc_hidden, config.hidden_size,
                                     bias=(True, True))

        self.entity = nn.Linear(config.hidden_size, 1)
        self.head2head = nn.Linear(config.hidden_size, 1)
        self.end2end = nn.Linear(config.hidden_size, 1)

    def forward(self, token_id, seg_id, bert_mask):

        outputs = self.bert_encoder(token_id, token_type_ids=seg_id, attention_mask=bert_mask)
        outputs = outputs[0]

        x_all_start = self.mlp_arc_start(outputs)
        x_all_end = self.mlp_arc_end(outputs)

        feature_logit_cond = self.ner_biaffine(x_all_start, x_all_end)
        # print(feature_logit_cond.shape)

        entity_logit = nn.Sigmoid()(self.entity(feature_logit_cond))
        head2head_logit = nn.Sigmoid()(self.head2head(feature_logit_cond))
        end2end_logit = nn.Sigmoid()(self.end2end(feature_logit_cond))
        #
        return entity_logit, head2head_logit, end2end_logit



class AliasDataset(Dataset):
    def __init__(self, document_list, tokenizer):
        super(AliasDataset, self).__init__()
        self.document_list = document_list
        self.tokenizer = tokenizer

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
            batch_entity_metrix = []
            batch_head2head_metrix = []
            batch_end2end_metrix = []
            batch_text = []
            batch_o_text = []
            batch_golden_answer = []

            local_max = 0
            for document in documents:
                text = document["sentence"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                batch_golden_answer.append(document["golden_answer"])
                text = document["sentence"]
                batch_text.append(text)
                batch_o_text.append(document["orient_sentence"])
                full_short_list = document["full_short_list"]
                text_word = [t for t in list(text)]
                codes = self.tokenizer.encode_plus(text_word,
                                                   return_offsets_mapping=True,
                                                   is_split_into_words=True,
                                                   max_length=local_max,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")

                input_ids_ = codes["input_ids"]
                iiv = len(input_ids_) - 1
                while input_ids_[iiv] == 0:
                    iiv -= 1
                # print(iiv + 1, document["sentence"], len(text_word), len(input_ids_), local_max)
                # print(input_ids_)
                assert iiv + 1 == len(document["sentence"]) + 2

                entity_metrix = torch.zeros((local_max, local_max))
                head2head_metrix = torch.zeros((local_max, local_max))
                end2end_metrix = torch.zeros((local_max, local_max))

                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                # print(full_short_list)
                for item in full_short_list:
                    full_s, full_e, short_s, short_e = item["key"]
                    entity_metrix[full_s + 1][full_e] = 1
                    entity_metrix[short_s + 1][short_e] = 1
                    head2head_metrix[full_s+1][short_s+1] = 1
                    end2end_metrix[full_e][short_e] = 1

                batch_entity_metrix.append(entity_metrix)
                batch_head2head_metrix.append(head2head_metrix)
                batch_end2end_metrix.append(end2end_metrix)

            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
            batch_entity_metrix = torch.stack(batch_entity_metrix, dim=0)
            batch_head2head_metrix = torch.stack(batch_head2head_metrix, dim=0)
            batch_end2end_metrix = torch.stack(batch_end2end_metrix, dim=0)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_entity_metrix": batch_entity_metrix,
                "batch_head2head_metrix": batch_head2head_metrix,
                "batch_end2end_metrix": batch_end2end_metrix,
                "batch_text": batch_text,
                "batch_o_text": batch_o_text,
                "batch_golden_answer": batch_golden_answer
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def evaluation(model, data_loader):
    hit_num = 0.0
    pred_num = 0.0
    true_num = 0.0
    model.eval()
    for idx, batch_data in enumerate(data_loader):
        entity_logit, head2head_logit, end2end_logit = model(
                batch_data["batch_input_ids"],
                batch_data["batch_token_type_id"],
                 batch_data["batch_attention_mask"])

        batch_size = entity_logit.shape[0]
        entity_logit = torch.where(entity_logit>0.5, 1.0, 0.0)

        for bi, entity_tag in enumerate(entity_logit):

            entity_res = []
            for xi in entity_tag.nonzero(as_tuple=False):
                xv = xi.cpu().numpy()
                entity_res.append(xv)
            print(entity_res)



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=50, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--mlp_arc_hidden', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_size', type=int, default=0, required=False)

    config = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    train_dataset = AliasDataset(train_list, tokenizer)

    data_loader = train_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    model = AliasModelV2(config)
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
    loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(data_loader):
            entity, head2head, end2end = model(
                batch_data["batch_input_ids"],
                batch_data["batch_token_type_id"],
                 batch_data["batch_attention_mask"])

            loss = loss_fn(entity.squeeze(-1), batch_data["batch_entity_metrix"])\
                   +loss_fn(head2head.squeeze(-1), batch_data["batch_head2head_metrix"])\
                   +loss_fn(end2end.squeeze(-1), batch_data["batch_end2end_metrix"])

            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                # loss_list.append(loss_value)

            loss.backward()
            optimizer.step()
            model.zero_grad()
        #     full_pred, short_pred = model(batch_data["batch_input_ids"],
        #               batch_data["batch_token_type_id"],
        #               batch_data["batch_attention_mask"],
        #                batch_data["batch_full_ids"])
        #
        #     # print(full_pred.shape)
        #     # print(short_pred.shape)
        #     #
        #     # print(batch_data["batch_full_label"].shape)
        #     # print(batch_data["batch_short_label"].shape)
        #     loss = 2*loss_fn(full_pred, batch_data["batch_full_label"]) + loss_fn(short_pred, batch_data["batch_short_label"])
        #     if idx % 10 == 0:
        #         loss_value = loss.data.cpu().numpy()
        #         print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
        #         # loss_list.append(loss_value)
        #
        #     loss.backward()
        #     optimizer.step()
        #     model.zero_grad()
        #
        evaluation(model, data_loader)
        # torch.save(model.state_dict(), "{}.pt".format("alias_model"))

if __name__ == "__main__":
    train()
