#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import torch
import argparse
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
from typing import List
from functools import partial
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from pytorch.layers.bert_optimization import BertAdam
from torch.utils.data import Dataset, DataLoader
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from torch.autograd import Variable
from pytorch.re.roberta_tplink import BertTplinkV2
from nlp_applications.ie_relation_extraction.evaluation import eval_metrix

data_path = "D:\data\关系抽取"
data_loader = LoaderDuie2Dataset(data_path, use_word_feature=False)
# triple_regularity = data_loader.triple_set
bert_model_name = "bert-base-chinese"

max_len = data_loader.max_seq_len
shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(max_len) for end_ind in range(ind, max_len)]
matrix_ind2shaking_ind = [[-1 for _ in range(max_len)] for _ in range(max_len)]
max_map = {i: 0 for i in range(1, max_len+1)}
for shaking_ind, matrix_ind in enumerate(shaking_ind2matrix_ind):
    matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind
    for k in max_map.keys():
        if k >= matrix_ind[0] and k >= matrix_ind[1]:
            max_map[k] = max(max_map[k], shaking_ind+1)



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


class Duie2Dataset(Dataset):
    def __init__(self, documents, bert_model_name, rel_num, is_train=True):
        super(Duie2Dataset, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        self.rel_num = rel_num
        self.is_train = is_train
        self.documents = documents

        num_added_toks = self.tokenizer.add_tokens(['[unused1]'])

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def _create_collate_fn(self, batch_first=False):

        def collate(documents: List[Document]):
            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_gold_answer = []

            batch_ent2ent_spans = []
            batch_head2head_spans = []
            batch_tail2tail_spans = []
            batch_token_offset = []
            batch_text = []

            batch_sequence_lens = []

            batch_size = len(documents)
            # print("hello inner v1")
            start = time.time()
            # local_max_len = 0
            # for document in documents:
            #     local_max_len = max(local_max_len, len(document.raw_text))

            for document in documents:
                # print("hello inner v2")
                gold_answer = []
                ent2ent_span = []
                head2head_span = []
                tail2tail_span = []

                text = document.raw_text
                text_word = ["[UNK]" if t in [" ", "\xa0", "\u3000", "�", "\ue5e5",
                                          "\u200e", "\u2003", "\x1c", "\uecd9", "\x7f", "\ue3e3",
                                          "\ufeff", "\uf440", "\uf46f",
                                          "\u200b", "\x05", "\x08", "\u200d",
                                          "\xad", "\uf760", "\uf78c", "\uf773",
                                          "\uf74b", "\uf792", "\uf762", "\uf771",
                                          "\uf6f3", "\uf794", "\uef80", "\ue007",
                                          "\uf701", "\uf731", "\uf700", "\uf717",
                                          "\uf739", "\uf734", "\uecd1", "\ue010",
                                          "\ue837", "\uecda", "\u200f", "\uf64f"] else t for t in list(text)]
                codes = self.tokenizer.encode_plus(text_word,
                             return_offsets_mapping=True,
                             add_special_tokens=False,
                             is_split_into_words=True,
                             max_length=max_len,
                             truncation=True,
                             return_length=True,
                             pad_to_max_length = True)
                input_ids = torch.tensor(codes["input_ids"]).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()
                offset_mapping = codes["offset_mapping"]



                # print(codes)
                seq_len = len(text_word)

                # print(seq_len, len(text_word), text_word)
                # print(input_ids, document.id)
                # print(text)
                assert seq_len == len(text_word)
                # print("hello inner v3")

                # print(len(text))
                # print(text)

                batch_text.append(text)
                # batch_token_offset.append(offset_mapping)
                batch_sequence_lens.append(seq_len)
                # print(codes["input_ids"])
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)

                for relation in document.relation_list:
                    s = relation.sub
                    p = relation.id
                    o = relation.obj

                    faker_s_start, faker_s_end = s.start, s.end-1
                    if faker_s_start == -1 or faker_s_end == -1:
                        print("error", s.start, s.end, [s.entity_text], document.id)
                        print(text)
                        # print(offset_mapping)
                        raise Exception("shake id is wrong, subject")

                    faker_o_start, faker_o_end = o.start, o.end-1
                    if faker_o_start == -1 or faker_o_end == -1:
                        print("error", o.start, o.end, [o.entity_text], document.id)
                        print(text)
                        print(offset_mapping)
                        raise Exception("shake id is wrong, object")

                    gold_answer.append((faker_s_start, faker_s_end, faker_o_start, faker_o_end, p))
                    ent2ent_span.append((faker_s_start,faker_s_end, 1))
                    ent2ent_span.append((faker_o_start, faker_o_end, 1))

                    if faker_s_start <= faker_o_start:
                        head2head_span.append((faker_s_start, faker_o_start, p, 1))
                    else:
                        head2head_span.append((faker_o_start, faker_s_start, p, 2))

                    if faker_s_end <= faker_o_end:
                        tail2tail_span.append((faker_s_end, faker_o_end, p, 1))
                    else:
                        tail2tail_span.append((faker_o_end, faker_s_end, p, 2))

                # print("hello inner v4")
                batch_ent2ent_spans.append(ent2ent_span)
                batch_head2head_spans.append(head2head_span)
                batch_tail2tail_spans.append(tail2tail_span)
                batch_gold_answer.append(gold_answer)


            if not self.is_train:
                batch_input_ids = torch.stack(batch_input_ids, dim=0)
                batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
                batch_token_type_id = torch.stack(batch_token_type_id, dim=0)

                return {
                    "batch_input_ids": batch_input_ids,
                    "batch_attention_mask": batch_attention_mask,
                    "batch_token_type_id": batch_token_type_id,
                    "batch_sequence_lens": batch_sequence_lens,
                    "batch_gold_answer": batch_gold_answer,
                }

            # print("hello inner v2 cost {}".format(time.time()-start))
            start = time.time()
            shaking_seq_len = max_len * (max_len + 1) // 2
            # shaking_seq_len = max_map[local_max_len]
            batch_ent2ent_seq_tag = torch.zeros(batch_size, shaking_seq_len).long()
            batch_ent2ent_seq_mask = torch.zeros(batch_size, shaking_seq_len).float()
            # batch_ent2ent_seq_mask[:, :seq_len] = 1

            for batch_id, spots in enumerate(batch_ent2ent_spans):
                seq_len = batch_sequence_lens[batch_id]
                # print(max_map[seq_len])
                batch_ent2ent_seq_mask[batch_id][:max_map[seq_len]] = 1
                # for iv in range(seq_len):
                #     # start = iv*max_len+iv
                #     # end = iv*max_len+seq_len
                #     # batch_ent2ent_seq_mask[batch_id][start:]
                #     for jv in range(iv, seq_len):
                #         shaking_ind = matrix_ind2shaking_ind[iv][jv]
                #         batch_ent2ent_seq_mask[batch_id][shaking_ind] = 1
                for sp in spots:
                    shaking_ind = matrix_ind2shaking_ind[sp[0]][sp[1]]
                    if shaking_ind == -1:
                        raise Exception("shake id is wrong")
                    tag_id = sp[2]
                    batch_ent2ent_seq_tag[batch_id][shaking_ind] = tag_id
                    # batch_ent2ent_seq_mask[batch_id][shaking_ind] = 2
            # print("hello inner v3 cost {}".format(time.time() - start))
            start = time.time()

            batch_head2head_seq_tag = torch.zeros(batch_size, self.rel_num, shaking_seq_len).long()
            batch_head2head_seq_mask = torch.zeros(batch_size, self.rel_num, shaking_seq_len).float()
            for batch_id, spots in enumerate(batch_head2head_spans):
                seq_len = batch_sequence_lens[batch_id]
                batch_head2head_seq_mask[batch_id,:,:max_map[seq_len]] = 1
                # for iv in range(seq_len):
                #     for jv in range(iv, seq_len):
                #         shaking_ind = matrix_ind2shaking_ind[iv][jv]
                #         for kv in range(self.rel_num):
                #             batch_head2head_seq_mask[batch_id][kv][shaking_ind] = 1
                for sp in spots:
                    shaking_ind = matrix_ind2shaking_ind[sp[0]][sp[1]]
                    if shaking_ind == -1:
                        raise Exception("shake id is wrong")
                    tag_id = sp[3]
                    rel_id = sp[2]
                    batch_head2head_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
                    # batch_head2head_seq_mask[batch_id][rel_id][shaking_ind] = 2
            # print("hello inner v4 cost {}".format(time.time() - start))
            start = time.time()
            batch_tail2tail_seq_tag = torch.zeros(batch_size, self.rel_num, shaking_seq_len).long()
            batch_tail2tail_seq_mask = torch.zeros(batch_size, self.rel_num, shaking_seq_len).float()
            for batch_id, spots in enumerate(batch_tail2tail_spans):
                seq_len = batch_sequence_lens[batch_id]
                batch_tail2tail_seq_mask[batch_id, :, :max_map[seq_len]] = 1
                for sp in spots:
                    shaking_ind = matrix_ind2shaking_ind[sp[0]][sp[1]]
                    if shaking_ind == -1:
                        print(sp[0], sp[1])
                        raise Exception("shake id is wrong")
                    tag_id = sp[3]
                    rel_id = sp[2]
                    batch_tail2tail_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
                # seq_len = batch_sequence_lens[batch_id]
                # for iv in range(seq_len):
                #     for jv in range(iv, seq_len):
                #         shaking_ind = matrix_ind2shaking_ind[iv][jv]
                #         for kv in range(self.rel_num):
                #             batch_tail2tail_seq_mask[batch_id][kv][shaking_ind] = 1

            # print("hello inner v5 cost {}".format(time.time() - start))
            start = time.time()
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
            # print("hello inner v6 cost {}".format(time.time() - start))

            # print("h2h shape", batch_head2head_seq_tag.shape)

            if self.is_train:
                return {
                    "batch_input_ids": batch_input_ids,
                    "batch_attention_mask": batch_attention_mask,
                    "batch_token_type_id": batch_token_type_id,
                    "batch_ent2ent_seq_tag": batch_ent2ent_seq_tag,
                    "batch_head2head_seq_tag": batch_head2head_seq_tag,
                    "batch_tail2tail_seq_tag": batch_tail2tail_seq_tag,
                    "batch_ent2ent_seq_mask": batch_ent2ent_seq_mask,
                    "batch_head2head_seq_mask": batch_head2head_seq_mask,
                    "batch_tail2tail_seq_mask": batch_tail2tail_seq_mask,
                    "batch_sequence_lens": batch_sequence_lens,
                    "batch_gold_answer": batch_gold_answer,
                    # "batch_text": batch_text,
                    # "batch_token_offset": batch_token_offset
                }
            else:
                return {
                    "batch_input_ids": batch_input_ids,
                    "batch_attention_mask": batch_attention_mask,
                    "batch_token_type_id": batch_token_type_id,
                    "batch_gold_answer": batch_gold_answer,
                    "batch_sequence_lens": batch_sequence_lens
                }


        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
import time
import sys

def get_sharing_spots_fr_shaking_tag(shaking_tag):
    spots = []

    for shaking_ind in shaking_tag.nonzero(as_tuple=False):
        shaking_ind_ = shaking_ind[0].item()
        tag_id = shaking_tag[shaking_ind_]
        matrix_inds = shaking_ind2matrix_ind[shaking_ind_]
        spot = (matrix_inds[0], matrix_inds[1], tag_id)
        spots.append(spot)
    return spots


def get_spots_fr_shaking_tag(shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero(as_tuple=False):
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

def extract_res(ent2ent_seq_tag, head2head_seq_tag, tail2tail_seq_tag, seq_len=0):

    entity_list = set()
    for xi in ent2ent_seq_tag.nonzero(as_tuple=False):
        xv = xi[0].cpu().numpy()
        start, end = shaking_ind2matrix_ind[xv]
        if end >= seq_len:
            continue
        entity_list.add((start, end))
    # print("entity_res", entity_list)

    h2h_list = []
    for xi in head2head_seq_tag.nonzero(as_tuple=False):
        xr = xi[0].cpu().numpy()
        xv = xi[1].cpu().numpy()
        x_tag = head2head_seq_tag[xr][xv].cpu().numpy()
        if x_tag == 1:
            s_start, o_start = shaking_ind2matrix_ind[xv]
            h2h_list.append((s_start, o_start, xr))
        elif x_tag == 2:
            o_start, s_start = shaking_ind2matrix_ind[xv]
            h2h_list.append((s_start, o_start, xr))

    # print(real_h2h)
    # print("h2h_list", h2h_list)

    t2t_list = []
    for xi in tail2tail_seq_tag.nonzero(as_tuple=False):
        xr = xi[0].cpu().numpy()
        xv = xi[1].cpu().numpy()
        x_tag = tail2tail_seq_tag[xr][xv].cpu().numpy()
        if x_tag == 1:
            s_end, o_end = shaking_ind2matrix_ind[xv]
            t2t_list.append((s_end, o_end, xr))
        elif x_tag == 2:
            o_end, s_end = shaking_ind2matrix_ind[xv]
            t2t_list.append((s_end, o_end, xr))

    # print("t2t_list", t2t_list)
    predict_list = []
    for ss, os, sr in h2h_list:
        for st, ot, tr in t2t_list:
            if (ss, st) not in entity_list:
                continue
            if (os, ot) not in entity_list:
                continue
            if sr != tr:
                continue
            predict_list.append((ss, st, os, ot, sr))
    return predict_list, entity_list


def eval_batch_data(model, batch_data, config=None):
    sub_entity_hit_num = 0.0
    sub_entity_pred_num = 0.0
    sub_entity_gold_num = 0.0

    sub_spo_hit_num = 0.0
    sub_spo_pred_num = 0.0
    sub_spo_gold_num = 0.0
    model.eval()
    with torch.no_grad():
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(Variable(batch_data["batch_input_ids"]).to(config.device),
                                                                                        Variable(batch_data["batch_attention_mask"]).to(config.device),
                                                                                        Variable(batch_data["batch_token_type_id"]).to(config.device))
        # batch_pred_ent_shaking_tag = torch.argmax(ent_shaking_outputs, dim=-1)
        # batch_pred_head_rel_shaking_tag = torch.argmax(head_rel_shaking_outputs, dim=-1)
        # batch_pred_tail_rel_shaking_tag = torch.argmax(tail_rel_shaking_outputs, dim=-1)
        batch_gold_answer = batch_data["batch_gold_answer"]
        batch_sequence_lens = batch_data["batch_sequence_lens"]
        for ind, gold_answer in enumerate(batch_gold_answer):

            entity_real_set = set()
            for g_ans in gold_answer:
                entity_real_set.add((g_ans[0], g_ans[1]))
                entity_real_set.add((g_ans[2], g_ans[3]))

            sub_entity_gold_num += len(entity_real_set)
            sub_spo_gold_num += len(gold_answer)

            seq_len = batch_sequence_lens[ind]

            ent_shaking_outputs_single = F.softmax(ent_shaking_outputs[ind], dim=-1)
            print(torch.max(ent_shaking_outputs_single[:,1]), "entity")

            head_rel_shaking_outputs_single = F.softmax(head_rel_shaking_outputs[ind], dim=-1)
            print(torch.max(head_rel_shaking_outputs_single[:, 1]), torch.max(head_rel_shaking_outputs_single[:, 2]), "head")

            tail_rel_shaking_outputs_single = F.softmax(tail_rel_shaking_outputs[ind], dim=-1)
            print(torch.max(tail_rel_shaking_outputs_single[:, 1]), torch.max(tail_rel_shaking_outputs_single[:, 2]), "tail")

            pred_ent_shaking_tag = torch.argmax(ent_shaking_outputs[ind], dim=-1)
            pred_head_rel_shaking_tag = torch.argmax(head_rel_shaking_outputs[ind], dim=-1)
            pred_tail_rel_shaking_tag = torch.argmax(tail_rel_shaking_outputs[ind], dim=-1)

            rel_list, entity_pre_list = extract_res(pred_ent_shaking_tag, pred_head_rel_shaking_tag, pred_tail_rel_shaking_tag, seq_len)

            sub_entity_pred_num += len(entity_pre_list)

            for s_entity in entity_pre_list:
                if s_entity in entity_real_set:
                    sub_entity_hit_num += 1

            sub_spo_pred_num += len(rel_list)
            for g_ans in gold_answer:
                if g_ans in rel_list:
                    sub_spo_hit_num += 1
    return {
        "sub_entity_hit_num": sub_entity_hit_num,
        "sub_entity_pred_num": sub_entity_pred_num,
        "sub_entity_gold_num": sub_entity_gold_num,
        "sub_spo_hit_num": sub_spo_hit_num,
        "sub_spo_pred_num": sub_spo_pred_num,
        "sub_spo_gold_num": sub_spo_gold_num
    }

def eval_data(model, data_loader, config):
    model.eval()

    start_time = time.time()
    entity_hit_num = 0.0
    entity_pred_num = 0.0
    entity_gold_num = 0.0

    spo_hit_num = 0.0
    spo_pred_num = 0.0
    spo_gold_num = 0.0
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            # batch_input_ids, batch_attention_mask, batch_token_type_id, batch_gold_answer = batch
            res = eval_batch_data(model, batch, config)
            entity_hit_num += res["sub_entity_hit_num"]
            entity_pred_num += res["sub_entity_pred_num"]
            entity_gold_num += res["sub_entity_gold_num"]
            spo_hit_num += res["sub_spo_hit_num"]
            spo_pred_num += res["sub_spo_pred_num"]
            spo_gold_num += res["sub_spo_gold_num"]

            print(res)

    entity_evaluation = eval_metrix(entity_hit_num, entity_gold_num, entity_hit_num)
    relation_evaluation = eval_metrix(spo_hit_num, spo_gold_num, spo_hit_num)
    final_res = {
        "entity_hit_num": entity_hit_num,
        "entity_pred_num": entity_pred_num,
        "entity_gold_num": entity_gold_num,
        "entity_recall": entity_evaluation["recall"],
        "entity_precision": entity_evaluation["precision"],
        "entity_f1_value": entity_evaluation["f1_value"],
        "spo_hit_num": spo_hit_num,
        "spo_pred_num": spo_pred_num,
        "spo_gold_num": spo_gold_num,
        "spo_recall": relation_evaluation["recall"],
        "spo_precision": relation_evaluation["precision"],
        "spo_f1_value": relation_evaluation["f1_value"],
    }
    print(final_res)
    with open("bert_tplink_{}.json".format(int(time.time())), "w") as f:
        f.write(json.dumps(final_res))



if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--hidden_size", type=int, default=768, required=False)
    parser.add_argument("--rel_size", type=int, default=len(data_loader.relation2id), required=False)
    parser.add_argument("--shaking_type", type=str, default="cln", required=False)
    parser.add_argument("--inner_enc_type", type=str, default="mix_pooling", required=False)
    parser.add_argument("--batch_size", type=int, default=5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    config = parser.parse_args()

    # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
    def bias_loss(weights=None):
        if weights is not None:
            weights = torch.FloatTensor(weights).to(device)
        cross_en = nn.CrossEntropyLoss(weight=weights)
        return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))


    loss_fct1 = bias_loss(torch.FloatTensor([1.0, 5.0]))
    loss_fct2 = bias_loss(torch.FloatTensor([1.0, 5.0, 5.0]))
    model = BertTplinkV2(config)
    model.to(device)

    dataset = Duie2Dataset(data_loader.documents, config.pretrain_name, config.rel_size)
    dev_dataset = Duie2Dataset(data_loader.dev_documents, config.pretrain_name, config.rel_size, is_train=False)

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    test_data_loader = dev_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    # for batch in test_data_loader:
    #     data = batch["batch_gold_answer"]
    #     for d in data:
    #         print(len(d))

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=config.learning_rate,
    #                      warmup=config.warmup_proportion,
    #                      t_total=int(len(dataset)//config.batch_size+1)*config.epoch)
    optimizer = torch.optim.Adamax(optimizer_grouped_parameters, lr=5e-5)
    step_gap = 10
    global_loss = 0.0
    total_steps = len(train_data_loader) * config.epoch + 1
    steps_per_ep = len(train_data_loader)
    z = (2 * config.rel_size + 1)
    path = "bert_tplink.model"
    for epoch in range(config.epoch):
        for step, batch in enumerate(train_data_loader):
            start = time.time()
            model.train()
            # print(len(batch))
             # + 1 avoid division by zero error
            # current_step = steps_per_ep * epoch + step
            # # w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            # # w_rel = min((config.rel_size / z) * current_step / total_steps, (config.rel_size / z))
            #
            w_ent = 2.0
            w_rel = 1.0
            # loss_weights = {"ent": w_ent, "rel": w_rel}
            # batch_input_ids, batch_attention_mask, batch_token_type_id, batch_ent2ent_seq_tag, batch_head2head_seq_tag, batch_tail2tail_seq_tag, batch_gold_answer = batch

            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(Variable(batch["batch_input_ids"]).to(device),
                                                                                            Variable(batch["batch_attention_mask"]).to(device),
                                                                                            Variable(batch["batch_token_type_id"]).to(device))

            #
            # # print(ent_shaking_outputs.shape, batch_ent2ent_seq_tag.shape)
            batch_ent2ent_seq_mask = torch.unsqueeze(batch["batch_ent2ent_seq_mask"], -1).to(device)
            ent_shaking_outputs *= batch_ent2ent_seq_mask
            ent_loss = loss_fct1(ent_shaking_outputs, Variable(batch["batch_ent2ent_seq_tag"]).to(device))
            #
            batch_head2head_seq_mask = torch.unsqueeze(batch["batch_head2head_seq_mask"], -1).to(device)

            head_rel_shaking_outputs *= batch_head2head_seq_mask
            head_loss = loss_fct2(head_rel_shaking_outputs, Variable(batch["batch_head2head_seq_tag"]).to(device))
            #
            batch_tail2tail_seq_mask = torch.unsqueeze(batch["batch_tail2tail_seq_mask"], -1)
            tail_rel_shaking_outputs *= batch_tail2tail_seq_mask
            tail_loss = loss_fct2(tail_rel_shaking_outputs, Variable(batch["batch_tail2tail_seq_tag"]).to(device))
            #
            loss = w_ent*ent_loss + w_rel*head_loss + w_rel*tail_loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            #
            # # if step % step_gap == 0:
            # #     global_loss += loss
            # #     if step == 0:
            # #         current_loss = global_loss
            # #     else:
            # #         current_loss = global_loss / step_gap
            # #     print(
            # #         u"step {} / {} of epoch {}, train/loss: {}".format(step, len(train_data_loader),
            # #                                                            epoch, current_loss))
            # #     # if step and step % 1000 == 0:
            # #     #     eval_data(model, dev_data_loader)
            # #     global_loss = 0.0
            cost_time = time.time()-start
            print(
                    u"step {0} / {1} of epoch {2}, train/loss: {3}, cost {4}".format(step, len(train_data_loader),
                                                                       epoch, loss, cost_time))
            if step and step % 1 == 0:

                eval_data(model, test_data_loader, config)
        # torch.save(model.state_dict(), path)
        break
