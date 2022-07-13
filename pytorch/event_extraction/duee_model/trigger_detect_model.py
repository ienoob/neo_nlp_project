#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import torch
import argparse
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from pytorch.layers.bert_optimization import BertAdam
from pytorch.layers.crf import CRF
from transformers import BertModel, BertTokenizer, BertTokenizerFast

bio2id = {
    "O": 0,
    "B": 1,
    "I": 2
}

filter_char = {" ", "\n", "\xa0", "\u3000", "\ue627"}

def get_train_dataset():
    path = "D:\\data\\句子级事件抽取\\duee_train.json\\duee_train.json"

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    train_dataset = []
    for item in data.split("\n"):
        if item.strip() == "":
            continue

        item_json = json.loads(item)
        bio_label = ["O"] * len(item_json["text"])
        assert len(item_json["text"]) <= 510
        for event in item_json["event_list"]:
            # item_json["text"][event["trigger_start_index"]:event["trigger_start_index"]
            bio_label[event["trigger_start_index"]] = "B"
            for ix in range(event["trigger_start_index"]+1, event["trigger_start_index"]+len(event["trigger"])):
                bio_label[ix] = "I"
            # print(event["trigger"], event["event_type"], event["trigger_start_index"])
        new_text = []
        new_label = []
        for i, x in enumerate(item_json["text"]):
            if x in filter_char:
                continue
            new_text.append(x)
            new_label.append(bio2id[bio_label[i]])
        train_dataset.append({
            "text": new_text,
            "label": new_label
        })
    return train_dataset


class DUEETriggerDataset(Dataset):

    def __init__(self, document_list, tokenizer):
        super(DUEETriggerDataset, self).__init__()
        self.qa_list = document_list
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
            batch_input_labels = []
            # batch_gold_answer = []

            local_max = 0
            for document in documents:
                text = document["text"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
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
                # print(text_word)

                # print(codes["offset_mapping"])
                iiv = len(input_ids_)-1
                while input_ids_[iiv]==0:
                    iiv -= 1
                # print(text_word)
                # print(iiv + 1, len(document["label"]), len(text_word), len(input_ids_), local_max)
                # print(input_ids_)
                assert iiv+1 == len(document["label"])+2
                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()
                # offset_mapping = codes["offset_mapping"]

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                label_id = [0] + document["label"]
                for _ in range(local_max-len(label_id)):
                    label_id.append(0)
                batch_input_labels.append(label_id)
                # batch_gold_answer.append(document[2])
            batch_input_labels = torch.LongTensor(batch_input_labels)
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).bool()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)


            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_input_labels": batch_input_labels,
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class BertCrf(nn.Module):

    def __init__(self, config):
        super(BertCrf, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        # print(type(self.bert_encoder))

        self.hidden2tag = nn.Linear(config.hidden_size, config.entity_size + 2)
        # self.drop = nn.Dropout(p=config.dropout)

        self.crf = CRF(config.entity_size, config.gpu)

        # self.register_parameter("weight_0_bert", self.bert_encoder.weight)
        # self.register_parameter("weight_0_crf", self.self.crf.weight)

    def forward(self, input_id, input_segment_id, input_mask, batch_label=None):
        outputs = self.bert_encoder(input_id, token_type_ids=input_segment_id, attention_mask=input_mask)
        outputs = outputs[0]
        tags = self.hidden2tag(outputs)
        # tags = self.drop(tags)

        if batch_label is None:
            scores, tag_seq = self.crf._viterbi_decode(tags, input_mask)
            return tag_seq
        else:
            total_loss = self.crf.neg_log_likelihood_loss(tags, input_mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(tags, input_mask)
            return total_loss, tag_seq


def train():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--entity_size', type=int, default=len(bio2id), required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--dropout', type=float, default=0.5, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    dataset = DUEETriggerDataset(get_train_dataset(), tokenizer)
    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    model = BertCrf(config)
    model.to(config.device)

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
                         t_total=int(len(dataset) // config.batch_size + 1) * config.epoch)
    # for idx, batch_data in enumerate(train_data_loader):
    #     pass
    # print("done")

    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(train_data_loader):
            loss, tag_seq = model(batch_data["batch_input_ids"].to(device),
                                  batch_data["batch_token_type_id"].to(device),
                                  batch_data["batch_attention_mask"].to(device),
                                  batch_data["batch_input_labels"].to(device))

            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))

            loss.backward()
            optimizer.step()
            model.zero_grad()


if __name__ == "__main__":
    train()
