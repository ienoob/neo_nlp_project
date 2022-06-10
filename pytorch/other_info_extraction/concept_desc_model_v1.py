#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    概念解释模型 v1
"""
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from functools import partial
from pytorch.layers.crf import CRF
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3

label2id = {"O": 0, "B-ENTITY": 1, "I-ENTITY": 2}


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.beta_dense = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.gamma_dense = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias

def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result)).to(data.device)


class ConceptDescDataset(Dataset):
    def __init__(self, sentence_list, tokenizer):
        super(ConceptDescDataset, self).__init__()
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, index):
        return self.sentence_list[index]

    def _create_collate_fn(self, batch_first=False):
        def collate(documents):
            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_input_labels = []
            batch_input_entity = []
            batch_is_desc_labels = []

            local_max = 0
            for document in documents:
                text = document["s_sentence"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                text = document["s_sentence"]
                codes = self.tokenizer.encode_plus(text,
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
                # print(iiv + 1, document["s_sentence"], len(text), len(input_ids_), local_max)
                # assert iiv + 1 == len(document["s_sentence"]) + 2

                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)

                label_id = [0] + document["s_label"]
                for _ in range(local_max - len(label_id)):
                    label_id.append(0)

                batch_input_labels.append(label_id)
                batch_input_entity.append(torch.LongTensor((document["s_entity"][0][0]+1, document["s_entity"][0][1])))
                batch_is_desc_labels.append(torch.LongTensor([document["is_desc"]]))

            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
            batch_input_entity = torch.stack(batch_input_entity, dim=0).long()
            batch_is_desc_labels = torch.stack(batch_is_desc_labels, dim=0).float()
            batch_input_labels = torch.LongTensor(batch_input_labels)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_input_entity": batch_input_entity,
                "batch_is_desc_labels": batch_is_desc_labels,
                "batch_input_labels": batch_input_labels
            }


        return partial(collate)
    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class ConceptDescModelV1(nn.Module):

    def __init__(self, config):
        super(ConceptDescModelV1, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        self.hidden2tag = nn.Linear(config.hidden_size, config.entity_bio_size + 2)
        self.crf = CRF(config.entity_bio_size, config.gpu)

        self.layer_norm = ConditionalLayerNorm(config.hidden_size, eps=1e-12)

        self.desc_span = nn.Linear(config.hidden_size, 1)
        self.id2label = {0: "O", 1: "B-ENTITY", 2: "I-ENTITY"}

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, batch_label=None, input_entity=None):
        outputs = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        outputs = outputs[0]

        tags = self.hidden2tag(outputs)

        if batch_label is None:
            scores, tag_seq = self.crf._viterbi_decode(tags, attention_mask)

            sentence_ids = []
            entity_span = []
            bert_encoders = []
            for ii, tag_pred in enumerate(tag_seq.detach().numpy()):
                tag_seq_list = [self.id2label.get(tag, "O") for tag in tag_pred]

                tag_value = extract_entity(tag_seq_list)

                for entity in tag_value:
                    sentence_ids.append(ii)
                    entity_span.append(torch.tensor((entity[0], entity[0] + entity[1] - 1), dtype=torch.long))
                    bert_encoders.append(outputs[ii])

            bert_encoders = torch.cat(bert_encoders).to(outputs.device)
            entity_span = torch.cat(entity_span).to(outputs.device)

            sub_start_encoder = batch_gather(bert_encoders, entity_span[:, 0])
            sub_end_encoder = batch_gather(bert_encoders, entity_span[:, 1])
            entity_feature = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(bert_encoders, entity_feature)

            desc_logits = nn.Sigmoid()(self.desc_span(context_encoder[:, 0]))

            return tag_seq, desc_logits, sentence_ids
        else:
            total_loss = self.crf.neg_log_likelihood_loss(tags, attention_mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(tags, attention_mask)

            sub_start_encoder = batch_gather(outputs, input_entity[:, 0])
            sub_end_encoder = batch_gather(outputs, input_entity[:, 1])

            entity_feature = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(outputs, entity_feature)

            desc_logits = nn.Sigmoid()(self.desc_span(context_encoder[:, 0]))

            return total_loss, tag_seq, desc_logits


def train():
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_bio_size', type=int, default=3, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    with open("D:\data\self-data\entity_desc_v1.json", "r", encoding="utf-8") as f:
        data = f.read()

    data_dict = json.loads(data)
    print(len(data_dict))
    data_list = []

    for dt in data_dict:
        s_entity = [(dt["entity_start"], dt["entity_start"]+len(dt["entity"]))]
        sentence = dt["sentence"]
        entity_label = ["O" for _ in sentence]
        entity_label[dt["entity_start"]] = "B-ENTITY"
        entity_label_id = [0 for _ in sentence]
        entity_label_id[dt["entity_start"]] = label2id["B-ENTITY"]
        for i in range(dt["entity_start"]+1, dt["entity_start"]+len(dt["entity"])):
            entity_label[i] = "I-ENTITY"
        sentence_text = []
        sentence_label = []
        for i, s in enumerate(sentence):
            if s in {" ", "\n", "\u2003", "\u200b", "\ufeff", "\xa0", "\u3000", "\x06", "\x07", "\u2002",
                     '¹', '²', "्"}:
                continue
            sentence_text.append(s)
            sentence_label.append(entity_label_id[i])

        data_list.append({"s_sentence": sentence_text,
                          "s_label": sentence_label,
                          "is_desc": 1,
                          "s_entity": s_entity})

    config = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    train_dataset = ConceptDescDataset(data_list, tokenizer)
    # print(config.batch_size)

    # res = tokenizer.encode_plus('హైదరాబాదు',return_offsets_mapping=True)
    # print(res)
    data_loader = train_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    model = ConceptDescModelV1(config)

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
        for idx, batch_data in enumerate(data_loader):
            total_loss, tag_seq, desc_logits = model(batch_data["batch_input_ids"],
                  batch_data["batch_token_type_id"],
                  batch_data["batch_attention_mask"],
                  batch_data["batch_input_labels"],
                  batch_data["batch_input_entity"]
                  )

            loss = total_loss + loss_fn(desc_logits, batch_data["batch_is_desc_labels"])
            if idx % 20 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))

            loss.backward()
            optimizer.step()
            model.zero_grad()

            torch.save(model.state_dict(), "D:\data\self-model\{}.pt".format("entity_desc"))

train()
