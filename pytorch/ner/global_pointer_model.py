#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import extract_entity, eval_metrix

msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")

label2id = {
    "no-entity": 0,
    "AGE": 1,
    "ANGLE": 2,
    "AREA": 3,
    "CAPACTITY": 4,
    "DATE": 5,
    "DECIMAL": 6,
    "DURATION": 7,
    "FRACTION": 8,
    "FREQUENCY": 9,
    "INTEGER": 10,
    "LENGTH": 11,
    "LOCATION": 12,
    "MEASURE": 13,
    "MONEY": 14,
    "ORDINAL": 15,
    "ORGANIZATION": 16,
    "PERCENT": 17,
    "PERSON": 18,
    "PHONE": 19,
    "POSTALCODE": 20,
    "RATE": 21,
    "SPEED": 22,
    "TEMPERATURE": 23,
    "TIME": 24,
    "WEIGHT": 25,
    "WWW": 26
}

def split(input_str, train=True):
    cut_list = []
    cut_index = []
    start = 0
    cache = []
    if train:

        for i, s in enumerate(input_str):
            cache.append(s)
            if cache.count("℃") % 2 ==0:
                if len(cache) > 128:
                    cut_list.append(cache)
                    cut_index.append((start, i+1))
                    cache = []
                    start = i+1
        if cache:
            cut_list.append(cache)
            cut_index.append((start, len(input_str)))
    else:
        for i, s in enumerate(input_str):
            cache.append(s)
            if s in ["）"]:
                if len(cache) > 128:
                    cut_list.append(cache)
                    cut_index.append((start, i+1))
                    cache = []
                    start = i+1
                # if len(cache)

        if cache:
            cut_list.append(cache)
            cut_index.append((start, len(input_str)))
    return cut_list, cut_index



def get_train_data(is_train=True):
    train_list = []
    if is_train:
        sentence_list = msra_data.train_sentence_list
        tag_list = msra_data.train_tag_list
    else:
        sentence_list = msra_data.test_sentence_list
        tag_list = msra_data.test_tag_list
    for i, item in enumerate(sentence_list):
        tag_data = tag_list[i]
        if len(item) > 510:
            # print(len(item))
            # print("".join(item))
            cut_list, cut_index = split(item, is_train)
            assert cut_index[-1][1] == len(item)
            for start, end in cut_index:
                sub_tag = tag_data[start:end]
                entity = [(st, ed, label2id[tp]) for st, ed, tp in extract_entity(sub_tag)]
                train_list.append({
                    "sentence": item[start:end],
                    "label": entity
                })
        else:
            entity = [(st, ed, label2id[tp]) for st, ed, tp in extract_entity(tag_data)]
            # print(entity)
            train_list.append({
                "sentence": item,
                "label": entity
            })
    return train_list



class MsraDataset(Dataset):

    def __init__(self, document_list, tokenizer):
        super(MsraDataset, self).__init__()
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
            batch_input_labels = []
            batch_golden_labels = []

            local_max = 0
            for document in documents:
                text = document["sentence"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                text_word = document["sentence"]
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
                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()
                # offset_mapping = codes["offset_mapping"]

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)

                entity_info = document["label"]
                label_metrix = torch.zeros((len(label2id) - 1, local_max, local_max))
                for st, ed, tp in entity_info:
                    label_metrix[tp-1][st+1][ed] = 1


                # label_id = [document["label"]]
                batch_input_labels.append(label_metrix)
                batch_golden_labels.append(entity_info)
                # batch_gold_answer.append(document[2])
            batch_input_labels = torch.stack(batch_input_labels, dim=0)
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_input_labels": batch_input_labels,
                "batch_golden_labels": batch_golden_labels
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


# get_train_data()

# for data in msra_data.train_tag_list:
#     print(data)

def evaluation(model, data_loader):
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0
    model.eval()
    for batch_data in data_loader:
        logits = model(batch_data["batch_input_ids"],
                       batch_data["batch_attention_mask"],
                       batch_data["batch_token_type_id"])

        batch_golden_labels = batch_data["batch_golden_labels"]

        for iv, item in enumerate(logits):
            golden_label = batch_golden_labels[iv]
            true_num += len(golden_label)

            nonzero = torch.nonzero(torch.greater(item, 0))
            for x, y, z in nonzero:
                x, y, z = int(x.detach().numpy())+1, int(y.detach().numpy())-1, int(z.detach().numpy())
                if y> z:
                    continue
                pre_num += 1

                if (x, y, z) in golden_label:
                    hit_num += 1

    print("pre {} true {} hit {}".format(pre_num, true_num, hit_num))
    print(eval_metrix(hit_num, pre_num, true_num))


import time
import argparse
from pytorch.layers.global_pointer import GlobalPointerV2
from transformers import BertModel, BertTokenizerFast

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--ent_type_size', type=int, default=len(label2id)-1, required=False)
    parser.add_argument('--inner_dim', type=int, default=64, required=False)
    parser.add_argument('--RoPE', type=bool, default=False, required=False)
    parser.add_argument('--scheduler', type=str, default="CAWR", required=False)  # Step
    parser.add_argument('--T_mult', type=int, default=1, required=False)
    parser.add_argument('--rewarm_epoch_num', type=int, default=2, required=False)
    parser.add_argument('--decay_rate', type=float, default=0.999, required=False)
    parser.add_argument('--decay_steps', type=int, default=100, required=False)

    config = parser.parse_args()

    # bert_encoder = BertModel.from_pretrained("bert-base-chinese")

    model = GlobalPointerV2(config)

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)
    train_data = get_train_data()
    eval_data = get_train_data(is_train=False)
    dataset = MsraDataset(train_data, tokenizer)
    train_data_loader = dataset.get_dataloader(16,
                                               shuffle=True,
                                               pin_memory=False)
    eval_dataset = MsraDataset(eval_data, tokenizer)
    eval_data_loader = eval_dataset.get_dataloader(16,
                                               shuffle=True,
                                               pin_memory=False)
    init_learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    if config.scheduler == "CAWR":
        T_mult = config.T_mult
        rewarm_epoch_num = config.rewarm_epoch_num
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_data_loader) * rewarm_epoch_num,
                                                                         T_mult)
    elif config.scheduler == "Step":
        decay_rate = config.decay_rate
        decay_steps = config.decay_steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        raise Exception
    start = time.time()
    epoch = 10
    for e in range(epoch):
        for b, batch_data in enumerate(train_data_loader):
            model.train()
            # print(batch_data["batch_input_ids"].shape)
            logits = model(batch_data["batch_input_ids"],
                                  batch_data["batch_attention_mask"],
                                  batch_data["batch_token_type_id"])
            loss = loss_fun(logits, batch_data["batch_input_labels"])
            if b % 10 == 0:
                print("epoch {} batch {} loss {} cost {}".format(e, b, loss.detach().numpy(), time.time()-start))
                start = time.time()
                #

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eval_start = time.time()
        evaluation(model, eval_data_loader)
        print("eval cost {}".format(time.time()-eval_start))
        # scheduler.step()
    # print(res.shape)
    # break


if __name__ == "__main__":
    # d = get_train_data()
    # print(d[0]["label"])
    train()
