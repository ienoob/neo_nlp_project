#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertTokenizerFast


# from transformers import RobertaModel, BertModel
from pytorch.layers.bert_optimization import BertAdam
from pytorch.syntactic_parsing.parser_layer import NonLinear, Biaffine
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3


msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")

sentence_length = []
sentence_length_v = []
def train_dataset():
    documents = []
    label2id = {"O": 0}
    for i, document in enumerate(msra_data.train_sentence_list):

        if len(document) > 500:
            # first_text = document[:300]
            # second_text = document[300:]
            label_seq = msra_data.train_tag_list[i]
            cut_point = 300

            while label_seq[cut_point] != "O":
                cut_point -= 1
            print("cut_point {}".format(cut_point))
            cut_list = [(0, cut_point), (cut_point, len(document))]

            for cut in cut_list:
                label_seq = msra_data.train_tag_list[i][cut[0]:cut[1]]
                entitys = extract_entity(label_seq)
                for entity in entitys:
                    if entity[2] not in label2id:
                        label2id[entity[2]] = len(label2id)
                documents.append({
                    "content": document[cut[0]:cut[1]],
                    "label": entitys
                })
                sentence_length.append(len(document[cut[0]:cut[1]]))
        else:
            entitys = extract_entity(msra_data.train_tag_list[i])
            for entity in entitys:
                # label_set.add(entity[2])
                if entity[2] not in label2id:
                    label2id[entity[2]] = len(label2id)
            documents.append({
                "content": document,
                "label": entitys
            })
            sentence_length.append(len(document))
    return documents, label2id


class MSADataset(Dataset):

    def __init__(self, document_list, tokenizer, label2id):
        super(MSADataset, self).__init__()
        self.document_list = document_list
        self.tokenizer = tokenizer
        self.label2id = label2id

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
                text = document["content"]
                local_max = max(len(text), local_max)
            local_max += 1


            for document in documents:
                label_metrix = torch.zeros((local_max, local_max))
                text_word = document["content"]
                codes = self.tokenizer.encode_plus(text_word,
                                                  return_offsets_mapping=True,
                                                  is_split_into_words=True,
                                                  max_length=local_max,
                                                  truncation=True,
                                                  return_length=True,
                                                  pad_to_max_length=True)
                input_ids = torch.tensor(codes["input_ids"]).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                # label_id = [0] + document[1]
                # for _ in range(local_max-len(label_id)):
                #     label_id.append(0)
                # batch_input_labels.append(label_id)
                for entity in document["label"]:
                    label_metrix[entity[0]+1][entity[1]] = self.label2id[entity[2]]
                batch_input_labels.append(label_metrix)
                batch_gold_answer.append(document["label"])
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


if __name__ == "__main__":
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
    train_data_list, label2id = train_dataset()

    df = pd.DataFrame({"s_len": sentence_length})

    print(df.describe())

    # plt.hist(x=sentence_length, bins=20,
    #          color="steelblue",
    #          edgecolor="black")
    # plt.show()
    config.entity_size = len(label2id)
    msa_train_dataset = MSADataset(train_data_list, tokenizer, label2id)

    msa_data_loader = msa_train_dataset.get_dataloader(config.batch_size,
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
                         t_total=int(len(msa_train_dataset) // config.batch_size + 1) * config.epoch)
    loss_list = []

    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(msa_data_loader):
            tag_logits = model(batch_data["batch_input_ids"],
                                  batch_data["batch_token_type_id"],
                                  batch_data["batch_attention_mask"])
            loss = compute_loss(tag_logits, batch_data["batch_input_labels"])

            if idx % 100 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                loss_list.append(loss_value)

            # print(loss.data)

            loss.backward()
            optimizer.step()
            model.zero_grad()
        plt.plot(loss_list)
        plt.show()
    #
        break

