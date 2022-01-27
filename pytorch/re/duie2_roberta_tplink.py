#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import argparse
import torch.nn.functional as F
from torch import nn
import numpy as np
import tensorflow as tf
from transformers import BertModel, BertTokenizer
from nlp_applications.ie_relation_extraction.evaluation import eval_metrix
from pytorch.layers.bert_optimization import BertAdam
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from pytorch.re.roberta_tplink import RobertaTplink

data_path = "D:\data\关系抽取"
data_loader = LoaderDuie2Dataset(data_path)
triple_regularity = data_loader.triple_set
bert_model_name = "hfl/chinese-roberta-wwm-ext"


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
    return torch.tensor(out_tensor)


class DataIter(BaseDataIterator):

    def __init__(self, input_loader):
        super(DataIter, self).__init__(input_loader)
        self.data_loader = input_loader
        self.token = BertTokenizer.from_pretrained(bert_model_name)

    def single_doc_processor(self, doc: Document):
        text_raw = doc.raw_text
        # print(text_raw)
        char_encode_id = self.token.encode(list(text_raw))
        # print(len(text_raw))
        # print(len(char_encode_id))

        entity_label_data = np.zeros((len(char_encode_id), len(char_encode_id)))
        hh_label_data = np.zeros((len(char_encode_id), len(char_encode_id)))
        tt_label_data = np.zeros((len(char_encode_id), len(char_encode_id)))
        mt_entity_mask = np.ones((len(char_encode_id), len(char_encode_id)))
        mt_entity_mask = np.triu(mt_entity_mask, k=0)
        mt_mask = np.ones((len(char_encode_id), len(char_encode_id)))

        entity_relation_value = []
        for relation in doc.relation_list:
            sub = relation.sub
            obj = relation.obj

            entity_label_data[sub.start+1][sub.end] = 1
            entity_label_data[obj.start+1][obj.end] = 1

            hh_label_data[sub.start+1][obj.start+1] = relation.id
            tt_label_data[sub.end][obj.end] = relation.id

            entity_relation_value.append((sub.start+1, sub.end, obj.start+1, obj.end, relation.id))

        return {
            "char_encode_id": char_encode_id,
            "entity_label_data": entity_label_data,
            "hh_label_data": hh_label_data,
            "tt_label_data": tt_label_data,
            "entity_relation_value": entity_relation_value,
            "text_raw": text_raw,
            "mt_mask": mt_mask,
            "mt_entity_mask": mt_entity_mask
        }

    def padding_batch_data(self, input_batch_data):
        batch_encode_id = []
        batch_entity_label = []
        batch_hh_label_data = []
        batch_tt_label_data = []
        batch_entity_relation_value = []
        batch_text_raw = []
        batch_mt_mask = []
        batch_mt_entity_mask = []

        max_len = 0
        for data in input_batch_data:
            batch_text_raw.append(data["text_raw"])
            batch_encode_id.append(data["char_encode_id"])
            batch_entity_relation_value.append(data["entity_relation_value"])

            max_len = max(len(data["char_encode_id"]), max_len)

        for data in input_batch_data:
            entity_label_data = data["entity_label_data"]
            # print(max_len, entity_label_data.shape)
            entity_label_data = np.pad(entity_label_data,
                              ((0,  max_len - entity_label_data.shape[0]), (0, max_len - entity_label_data.shape[1])),
                              'constant', constant_values=0)

            batch_entity_label.append(entity_label_data)
            hh_label_data = data["hh_label_data"]
            hh_label_data = np.pad(hh_label_data,
                                       ((0, max_len - hh_label_data.shape[0]),
                                        (0, max_len - hh_label_data.shape[1])),
                                       'constant', constant_values=0)
            batch_hh_label_data.append(hh_label_data)

            tt_label_data = data["tt_label_data"]
            tt_label_data = np.pad(tt_label_data,
                                   ((0, max_len - tt_label_data.shape[0]),
                                    (0, max_len - tt_label_data.shape[1])),
                                   'constant', constant_values=0)
            batch_tt_label_data.append(tt_label_data)

            mt_mask = data["mt_mask"]
            mt_mask = np.pad(mt_mask,
                                   ((0, max_len - mt_mask.shape[0]),
                                    (0, max_len - mt_mask.shape[1])),
                                   'constant', constant_values=0)
            batch_mt_mask.append(mt_mask)

            mt_entity_mask = data["mt_entity_mask"]
            mt_entity_mask = np.pad(mt_entity_mask,
                             ((0, max_len - mt_entity_mask.shape[0]),
                              (0, max_len - mt_entity_mask.shape[1])),
                             'constant', constant_values=0)
            batch_mt_entity_mask.append(mt_entity_mask)

        batch_char_encode_id = sequence_padding(batch_encode_id, max_len)
        batch_char_encode_id = torch.LongTensor(batch_char_encode_id)

        return {
            "char_encode_id": batch_char_encode_id,
            "entity_label_data": torch.FloatTensor(batch_entity_label),
            "hh_label_data": torch.LongTensor(batch_hh_label_data),
            "tt_label_data": torch.LongTensor(batch_tt_label_data),
            "entity_relation_value": batch_entity_relation_value,
            "text_raw": batch_text_raw,
            "max_len": torch.LongTensor(max_len),
            "mt_mask": torch.ByteTensor(batch_mt_mask),
            "mt_entity_mask": torch.ByteTensor(batch_mt_entity_mask)
        }

    def dev_iter(self, input_batch_num):
        c_batch_data = []
        for doc in self.data_loader.dev_documents:
            c_batch_data.append(self.single_doc_processor(doc))
            if len(c_batch_data) == input_batch_num:
                yield self.padding_batch_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_batch_data(c_batch_data)

def multiclass_loss1(selection_mask, selection_logits, selection_gold):
    selection_gold = selection_mask * selection_gold
    selection_logits *=  selection_mask
    # selection_gold = selection_gold.unsqueeze(-1)
    # selection_mask = selection_mask.unsqueeze(-1)
    b, l1, l1, l2 = selection_logits.shape
    selection_loss = F.cross_entropy(selection_logits.view(b * l1*l1, l2), selection_gold.view(b * l1*l1))
    # selection_loss = selection_loss.masked_select(selection_mask).sum()
    return selection_loss

def multiclass_loss2(selection_mask, selection_logits, selection_gold):
    selection_gold = selection_mask * selection_gold
    selection_logits *= selection_mask
    # selection_gold = selection_gold.unsqueeze(-1)
    # selection_mask = selection_mask.unsqueeze(-1)
    b, l1, l1, l2 = selection_logits.shape
    selection_loss = F.cross_entropy(selection_logits.view(b * l1*l1, l2), selection_gold.view(b * l1*l1))
    # selection_loss = selection_loss.masked_select(selection_mask).sum()
    return selection_loss


def bce_loss(selection_mask, selection_logits, selection_gold):
    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
    selection_gold = selection_gold*selection_mask
    selection_logits = selection_logits*selection_mask
    # selection_mask = selection_mask.unsqueeze(-1)
    # print(selection_gold.shape)
    # print(selection_logits.shape)
    selection_gold = selection_gold.unsqueeze(-1)
    # selection_gold = torch.where(torch.greater(selection_gold, 0), 5.0, 1.0) * selection_gold
    selection_loss = loss_fct(selection_logits, selection_gold)
    # print(selection_mask.shape)
    # selection_loss = selection_loss.masked_select(selection_mask).sum()
    # selection_loss /= selection_mask.sum()
    return selection_loss


def evaluation(batch_data, model):
    model.eval()
    hit_num = 0.0
    true_num = 0.0
    predict_num = 0.0
    entity_logits, hh_logits, tt_logits = model(batch_data["char_encode_id"])
    entity_logits = nn.Sigmoid()(entity_logits)
    # print(tf.reduce_max(entity_logits))
    mt_mask = batch_data["mt_mask"].numpy()
    print(torch.max(entity_logits), "max value item")

    entity_argmax = torch.greater(entity_logits, 0.5).squeeze(-1).long().numpy()
    t_batch_num, l, l = entity_argmax.shape
    entity_argmax = entity_argmax * mt_mask
    hh_argmax = torch.argmax(hh_logits, dim=-1).numpy()
    hh_argmax = hh_argmax * mt_mask
    tt_argmax = torch.argmax(tt_logits, dim=-1).numpy()
    tt_argmax = tt_argmax * mt_mask


    for i in range(t_batch_num):
        true_predict = set(batch_data["entity_relation_value"][i])
        true_num += len(true_predict)

        entity_label_multi = entity_argmax[i]
        entity_list = set()
        for iv, srow in enumerate(entity_label_multi):
            for jv, ei in enumerate(srow):
                if jv < iv:
                    continue
                if ei == 1:
                    entity_list.add((iv, jv))
        print("entity", len(entity_list))
        hh_dict = dict()
        hh_rel_multi = hh_argmax[i]
        print(tf.reduce_max(hh_rel_multi))
        for iv, hrow in enumerate(hh_rel_multi):
            for jv, ei in enumerate(hrow):
                if ei == 0:
                    continue
                hh_dict.setdefault(ei, [])
                hh_dict[ei].append((iv, jv))

        print(len(hh_dict))
        tt_dict = dict()
        tt_rel_multi = tt_argmax[i]
        for iv, trow in enumerate(tt_rel_multi):
            for jv, ei in enumerate(trow):
                if ei == 0:
                    continue
                tt_dict.setdefault(ei, [])
                tt_dict[ei].append((iv, jv))
        print(len(tt_dict))

        predict_extract = []
        for kr, h_list in hh_dict.items():
            if kr not in tt_dict:
                continue
            tr_list = tt_dict[kr]
            for hs, ho in h_list:
                for ts, to in tr_list:
                    if (hs, ts) not in entity_list:
                        continue
                    if (ho, to) not in entity_list:
                        continue
                    p_value = (hs, ts, ho, to, kr)
                    predict_extract.append(p_value)
                    predict_num += 1
                    if p_value in true_predict:
                        hit_num += 1

    return {
        "hit_num": hit_num,
        "real_count": true_num,
        "predict_count":  predict_num
    }


if __name__ == "__main__":
    data_iter = DataIter(data_loader)
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--rel_num", type=int, default=len(data_loader.relation2id), required=False)
    parser.add_argument("--rel_emb_size", type=int, default=256, required=False)
    parser.add_argument("--decay", type=float, default=.75, required=False)
    parser.add_argument("--warmup_proportion", type=float, default=0.1, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--hidden_size", type=int, default=768, required=False)
    parser.add_argument("--eh_size", type=int, default=50, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)

    config = parser.parse_args()

    model = RobertaTplink(config)

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
                         t_total=1000)

    for i, batch_data in enumerate(data_iter.train_iter(config.batch_size)):
        model.train()
        a, b, c = model(batch_data["char_encode_id"])

        loss1 = bce_loss(batch_data["mt_entity_mask"], a, batch_data["entity_label_data"])
        # print(loss1.item())


        loss2 = multiclass_loss1(batch_data["mt_mask"], b, batch_data["hh_label_data"])
        # print(loss2.item())


        loss3 = multiclass_loss2(batch_data["mt_mask"], c, batch_data["tt_label_data"])

        # print(loss3.item())

        loss = loss1 + loss2 + loss3

        loss.backward()

        if i % 10 == 0:
            res = evaluation(batch_data, model)
            print("batch num {0} loss value is {1} eval {2}".format(i, loss.item(), res))
        # print(loss.item())
        # loss = loss.item()
        optimizer.step()
        optimizer.zero_grad()

