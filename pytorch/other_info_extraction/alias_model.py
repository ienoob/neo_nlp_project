#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
# import hanlp
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from pytorch.other_info_extraction.get_alias_data_v3 import train_list
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3


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
            batch_full_ids = []
            batch_full_metrix = []
            batch_short_metrix = []
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

                full_metrix = torch.zeros((local_max, 2))
                short_metrix = torch.zeros((local_max, 2))

                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                # print(full_short_list)
                for item in full_short_list:
                    full_s, full_e, _, _ = item["key"]
                    full_metrix[full_s+1][0] = 1
                    full_metrix[full_e][1] = 1

                # print("full item {}".format(len(full_short_list)))
                if len(full_short_list):
                    item = np.random.choice(full_short_list)
                    # if len(full_short_list) > 1:
                    #     print(full_short_list)
                    #     print(item)
                    # print(item)

                    full_s, full_e, short_s, short_e = item["key"]
                    batch_full_ids.append((full_s + 1, full_e))
                    short_metrix[short_s+1][0] = 1
                    short_metrix[short_e][1] = 1
                else:
                    batch_full_ids.append((0, 0))
                    short_metrix[0][0] = 1
                    short_metrix[0][1] = 1
                batch_full_metrix.append(full_metrix)
                batch_short_metrix.append(short_metrix)

            batch_full_ids = torch.tensor(batch_full_ids)
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
            batch_full_label = torch.stack(batch_full_metrix, dim=0)
            batch_short_label = torch.stack(batch_short_metrix, dim=0)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_full_label": batch_full_label,
                "batch_short_label": batch_short_label,
                "batch_full_ids": batch_full_ids,
                "batch_text": batch_text,
                "batch_o_text": batch_o_text,
                "batch_golden_answer": batch_golden_answer
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result)).to(data.device)


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


class AliasModel(nn.Module):

    def __init__(self, config):
        super(AliasModel, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        self.layer_norm = ConditionalLayerNorm(config.hidden_size, eps=1e-12)
        self.full = nn.Linear(config.hidden_size, 2)
        self.f_activate = nn.Sigmoid()
        self.short = nn.Linear(config.hidden_size, 2)
        self.s_activate = nn.Sigmoid()

    def forward(self, token_id, seg_id, bert_mask, input_full=None, is_train=True):
        outputs = self.bert_encoder(token_id, token_type_ids=seg_id, attention_mask=bert_mask)
        outputs = outputs[0]
        # print(outputs.shape)

        if is_train:
            sub_start_encoder = batch_gather(outputs, input_full[:, 0])
            sub_end_encoder = batch_gather(outputs, input_full[:, 1])

            full_name = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(outputs, full_name)

            full_pred = self.full(outputs).reshape(token_id.size(0), -1, 2)
            full_pred = self.f_activate(full_pred)

            short_pred = self.short(context_encoder).reshape(token_id.size(0), -1, 2)
            short_pred = self.s_activate(short_pred)

            return full_pred, short_pred
        else:
            seq_lens = bert_mask.sum(axis=-1)
            # print("seq_lens", seq_lens)
            full_preds = self.f_activate((self.full(outputs)))
            answer_list = list()

            for ii, full_pred in enumerate(full_preds.detach().numpy()):
                start = np.where(full_pred[:, 0] > 0.6)[0]
                end = np.where(full_pred[:, 1] > 0.5)[0]
                subjects = []
                for i in start:
                    j = end[end >= i]
                    if i == 0 or i > seq_lens[ii]:
                        continue

                    if len(j) > 0:
                        j = j[0]
                        if j > seq_lens[ii] - 2:
                            continue
                        subjects.append((i, j))

                answer_list.append(subjects)
            # print(len(answer_list))
            #
            sentence_ids = []
            bert_encoders, pass_ids, subject_ids, token_type_ids = [], [], [], []
            for ii, subjects in enumerate(answer_list):
                # print(ii, subjects)
                if subjects:
                    # pass_tensor = input_x[ii, :].unsqueeze(0).expand(len(subjects), input_x.size(1))
                    new_bert_encoder = outputs[ii, :, :].unsqueeze(0).expand(len(subjects), outputs.size(1),
                                                                                 outputs.size(2))
            #
                    token_type_id = torch.zeros((len(subjects), token_id.size(1)), dtype=torch.long)
                    for index, (start, end) in enumerate(subjects):
                        token_type_id[index, start:end + 1] = 1
                        sentence_ids.append(ii)
                    # pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    bert_encoders.append(new_bert_encoder)
            #
            # sentence_ids = torch.cat(sentence_ids).to(outputs.device)
            # pass_ids = torch.cat(pass_ids).to(outputs.device)
            if len(bert_encoders) == 0:
                return [], [], []

            bert_encoders = torch.cat(bert_encoders).to(outputs.device)
            subject_ids = torch.cat(subject_ids).to(outputs.device)
            #

            sub_start_encoder = batch_gather(bert_encoders, subject_ids[:, 0])
            sub_end_encoder = batch_gather(bert_encoders, subject_ids[:, 1])
            subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(bert_encoders, subject)

            po_tensor = self.short(context_encoder)
            # print(po_pred.shape)
            po_tensor = po_tensor.reshape(subject_ids.size(0), -1, 2)
            # print(po_tensor.shape, "++")
            po_tensor = nn.Sigmoid()(po_tensor)

            return sentence_ids, subject_ids, po_tensor



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=50, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
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

    model = AliasModel(config)
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
            full_pred, short_pred = model(batch_data["batch_input_ids"],
                      batch_data["batch_token_type_id"],
                      batch_data["batch_attention_mask"],
                       batch_data["batch_full_ids"])

            # print(full_pred.shape)
            # print(short_pred.shape)
            #
            # print(batch_data["batch_full_label"].shape)
            # print(batch_data["batch_short_label"].shape)
            loss = 2*loss_fn(full_pred, batch_data["batch_full_label"]) + loss_fn(short_pred, batch_data["batch_short_label"])
            if idx % 10 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))
                # loss_list.append(loss_value)

            loss.backward()
            optimizer.step()
            model.zero_grad()

        evaluation(model, data_loader)
        torch.save(model.state_dict(), "{}.pt".format("alias_model"))
        # break

# HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
# res = HanLP(['集微网消息，工信部国家级创新中心国汽智联、自然资源部直属企业中国地图出版集团以及四维图新等单位近日共同成立了国汽智图（北京）科技有限公司（简称：国汽智图）'])
#
# print(res["ner/ontonotes"])

def evaluation(model, data_loader):
    hit_num = 0.0
    pred_num = 0.0
    true_num = 0.0
    model.eval()
    for idx, batch_data in enumerate(data_loader):

        sentence_ids, subject_ids, po_tensor = model(batch_data["batch_input_ids"],
                                                     batch_data["batch_token_type_id"],
                                                     batch_data["batch_attention_mask"],
                                                     is_train=False)
        batch_res = [[] for _ in batch_data["batch_golden_answer"]]
        for ii, sid in enumerate(sentence_ids):
            start = subject_ids[ii].cpu().numpy()[0]
            end = subject_ids[ii].cpu().numpy()[1]
            # print(batch_data["batch_o_text"][sid])
            full_name = "".join(batch_data["batch_text"][sid][start - 1:end])
            s_len = len(batch_data["batch_text"][sid]) + 2

            po_res = po_tensor[ii].detach().numpy()

            start = np.where(po_res[:, 0] > 0.5)[0]
            end = np.where(po_res[:, 1] > 0.5)[0]
            objects = []
            for iv in start:
                jv = end[end >= iv]
                if iv == 0 or iv > s_len:
                    continue

                if len(jv) > 0:
                    jv = jv[0]
                    if jv > s_len - 2:
                        continue
                    objects.append((iv, jv))

            for obi, obj in objects:
                short_name = "".join(batch_data["batch_text"][sid][obi - 1:obj])

                batch_res[sid].append((full_name, short_name))

        for ii, s_res in enumerate(batch_res):
            pred_num += len(s_res)
            true_num += len(batch_data["batch_golden_answer"][ii])
            # print(s_res)
            # print(batch_data["batch_golden_answer"][ii])

            for item in s_res:
                # print(item)
                # print(batch_data["batch_golden_answer"][ii])
                if item in batch_data["batch_golden_answer"][ii]:
                    hit_num += 1

    print("pred_num: {}".format(pred_num))
    print("true_num: {}".format(true_num))
    print("hit_num: {}".format(hit_num))
    print(eval_metrix_v3(hit_num, true_num, pred_num))


def save_load_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_size', type=int, default=0, required=False)

    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)
    model = AliasModel(config)
    model_pt = torch.load("alias_model.pt")
    model.load_state_dict(model_pt)


    train_dataset = AliasDataset(train_list, tokenizer)
    data_loader = train_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    evaluation(model, data_loader)
    # for idx, batch_data in enumerate(data_loader):
    #     model.eval()
    #     sentence_ids, subject_ids, po_tensor = model(batch_data["batch_input_ids"],
    #                                   batch_data["batch_token_type_id"],
    #                                   batch_data["batch_attention_mask"],
    #                                   is_train=False)
    #
    #     # print(po_tensor.shape, "xxxx")
    #     for ii, sid in enumerate(sentence_ids):
    #         print(sid)
    #         print(subject_ids[ii])
    #         start = subject_ids[ii].cpu().numpy()[0]
    #         end = subject_ids[ii].cpu().numpy()[1]
    #         print(batch_data["batch_o_text"][sid])
    #         print("".join(batch_data["batch_text"][sid][start-1:end]))
    #         s_len = len(batch_data["batch_text"][sid])+2
    #
    #         po_res = po_tensor[ii].detach().numpy()
    #         print(po_res.shape, "++++++++++")
    #
    #         start = np.where(po_res[:,0] > 0.6)[0]
    #         end = np.where(po_res[:,1] > 0.5)[0]
    #         objects = []
    #         for iv in start:
    #             jv = end[end >= iv]
    #             if iv == 0 or iv > s_len:
    #                 continue
    #
    #             if len(jv) > 0:
    #                 jv = jv[0]
    #                 if jv > s_len - 2:
    #                     continue
    #                 objects.append((iv, jv))
    #
    #         for obi, obj in objects:
    #             print("".join(batch_data["batch_text"][sid][obi-1:obj]))
    #     #
    #     #
    #     # break

def get_input_data(tokenizer, input_sentence):
    local_max = len(input_sentence)
    text_word = [t for t in input_sentence if t not in [" "]]
    codes = tokenizer.encode_plus(text_word,
                                   return_offsets_mapping=True,
                                   is_split_into_words=True,
                                   max_length=local_max,
                                   truncation=True,
                                   return_length=True,
                                   padding="max_length")

    input_ids_ = codes["input_ids"]
    input_ids = torch.tensor(input_ids_).long()
    attention_mask = torch.tensor(codes["attention_mask"]).long()
    token_type_ids = torch.tensor(codes["token_type_ids"]).long()

    batch_input_ids = torch.stack([input_ids], dim=0)
    batch_attention_mask = torch.stack([attention_mask], dim=0).byte()
    batch_token_type_id = torch.stack([token_type_ids], dim=0)

    iv = 0
    jv = 0
    dv = dict()
    for s in input_sentence:
        if s not in [" "]:
            dv[iv] = jv
            iv += 1
        jv += 1

    return {
        "batch_input_ids": batch_input_ids,
        "batch_attention_mask": batch_attention_mask,
        "batch_token_type_id": batch_token_type_id,
        "batch_text": text_word,
        "offset": dv
    }


def get_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_size', type=int, default=0, required=False)

    config = parser.parse_args()
    model = AliasModel(config)
    model_pt = torch.load("alias_model.pt")
    model.load_state_dict(model_pt)

    return model

def extractor(input_sentence, model=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_size', type=int, default=0, required=False)

    config = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    if model is None:
        model = AliasModel(config)
        model_pt = torch.load("alias_model.pt")
        model.load_state_dict(model_pt)

    p_data = get_input_data(tokenizer, input_sentence)

    offset = p_data["offset"]

    sentence_ids, subject_ids, po_tensor = model(p_data["batch_input_ids"],
                                                 p_data["batch_token_type_id"],
                                                 p_data["batch_attention_mask"],
                                                 is_train=False)

    s_len = len(input_sentence)
    # print(s_len)\\
    ans = []
    ans_full = []
    for ii, sid in enumerate(sentence_ids):
        print(sid)
        print(subject_ids[ii])
        start = subject_ids[ii].cpu().numpy()[0]
        end = subject_ids[ii].cpu().numpy()[1]
        # print(batch_data["batch_o_text"][sid])
        print(start, end, "---------")
        print("".join(p_data["batch_text"]))
        subject = "".join(p_data["batch_text"][start - 1:end])
        print("subject", subject)
        # s_len = len(batch_data["batch_text"][sid]) + 2

        po_res = po_tensor[ii].detach().numpy()
        # print(po_res.shape, "++++++++++")

        start_po = np.where(po_res[:, 0] > 0.6)[0]
        end_po = np.where(po_res[:, 1] > 0.5)[0]
        objects = []
        for iv in start_po:
            jv = end_po[end_po >= iv]
            if iv == 0 or iv > s_len:
                continue

            if len(jv) > 0:
                jv = jv[0]
                if jv > s_len - 2:
                    continue
                objects.append((iv, jv))

        for obi, obj in objects:
            object = "".join(p_data["batch_text"][obi - 1:obj])
            print("object", object)
            ans.append("{}->{}".format(subject, object))
            print(input_sentence[offset[start - 1]:offset[end-1]+1])
            print(input_sentence[offset[obi - 1]:offset[obj-1] + 1])
            ans_full.append({
                "name": input_sentence[offset[start - 1]:offset[end-1]+1],
                "alias": input_sentence[offset[obi - 1]:offset[obj-1] + 1],
                "name_idx": [offset[start - 1], offset[end-1]],
                "alias_idx": [offset[obi - 1], offset[obj-1]]
            })

    return ans, ans_full


if __name__ == "__main__":
    # import pandas as pd
    #
    # model = get_model()
    # #
    # data_path = "D:\data\\alias_sentence(2).csv"
    #
    # data = pd.read_csv(data_path)
    # n_df = []
    # iv = 0
    # for idx, dt in data.iterrows():
    #     if len(dt["sentence"])>500:
    #         continue
    #     # if iv > 20:
    #     #     break
    #     iv += 1
    #     e_ans, e_ans_full = extractor(dt["sentence"], model)
    #     n_df.append({"sentence": dt["sentence"], "ans": ":".join(e_ans), "true": 1,  "ans_idx": json.dumps(e_ans_full, ensure_ascii=False)})
    #     print("{}/{} complete".format(idx, data.shape[0]))
    #
    # n_df = pd.DataFrame(n_df)
    #
    # n_df.to_csv("alias_res.csv", index=False)

    # df = pd.read_csv("alias_res.csv", encoding="utf-8")
    # df.to_csv("alias_gbk.csv", index=False, encoding="gbk")


    train()
