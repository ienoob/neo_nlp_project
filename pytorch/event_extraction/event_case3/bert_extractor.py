#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import torch
import json
import argparse
import hanlp
import pandas as pd
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from pytorch.event_extraction.event_case3.train_data import rt_data
from pytorch.event_extraction.event_model_v1 import EventModelV1
from nlp_applications.ner.evaluation import eval_metrix, extract_entity
from pytorch.event_extraction.bert_utils import split_cha
from pytorch.event_extraction.crf_utils import DSU
label2id = rt_data["label2id"]
role2id = rt_data["role2id"]
# bert_train_list = rt_data["bert_data"]["train"]

id2label = {v: k for k, v in label2id.items()}
id2role = {v: k for k, v in role2id.items()}


parser = argparse.ArgumentParser(description="")

parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
parser.add_argument("--batch_size", type=int, default=16, required=False)
parser.add_argument("--epoch", type=int, default=30, required=False)
parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
parser.add_argument("--shuffle", type=bool, default=True, required=False)
parser.add_argument('--pin_memory', type=bool, default=False, required=False)
parser.add_argument('--hidden_size', type=int, default=768, required=False)
parser.add_argument('--entity_size', type=int, default=len(label2id), required=False)
parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
parser.add_argument('--dropout', type=float, default=0.5, required=False)
parser.add_argument('--gpu', type=bool, default=False, required=False)
config = parser.parse_args()

tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)
bert_model = EventModelV1(config)
model_pt = torch.load("all-bidding.pt")
bert_model.load_state_dict(model_pt)


def cut_text(input_text):
    text_sentence = input_text.split("\n")
    text_cut = []
    text_cache = []
    cache_len = 0

    if len(text_sentence) == 1:
        if len(text_sentence[0]) < 500:
            text_cut.append(text_sentence[0])
        else:
            text_sentence = text_sentence[0].split("。")
            for sentence in text_sentence:
                if cache_len + len(text_cache) - 1 + len(sentence) > 500:
                    text_cut.append("。".join(text_cache))
                    cache_len = 0
                    text_cache = []
                text_cache.append(sentence)
                cache_len += len(sentence)
            if text_cache:
                text_cut.append("。".join(text_cache))
    else:
        for sentence in text_sentence:
            if len(sentence) > 500:
                print("error")
                continue
            if cache_len + len(text_cache) - 1 + len(sentence) > 500:
                text_cut.append("\n".join(text_cache))
                cache_len = 0
                text_cache = []

            text_cache.append(sentence)
            cache_len += len(sentence)
        if text_cache:
            text_cut.append("\n".join(text_cache))

    # for text in text_cut:
    #     print(len(text)<=500)
    return text_cut


def bert2extract_ner(input_text):

    tempt_res = []
    text_list = cut_text(input_text)

    # 文章时间
    # pattern = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}"
    # datetimes = re.findall(pattern, input_text)
    # print("时间")
    # print(datetimes)


    for input_text in text_list:
        split_word = []
        cut_point = []
        # print(len(input_text))
        offset = dict()
        ci = 0
        for iv, t in enumerate(input_text):
            if t == "\n":
                if len(cut_point) == 0 or (cut_point and cut_point[-1] != ci):
                    cut_point.append(ci)

            elif t == "。":
                if len(cut_point) == 0 or (cut_point and cut_point[-1] != ci+1):
                    cut_point.append(ci+1)
            if t in split_cha:
                continue
            offset[ci] = iv
            ci += 1
            split_word.append(t)
        offset[ci] = len(input_text)
        cut_point.append(len(split_word))
        print(cut_point)
        batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
        input_text_ = "".join(split_word)
        text_word = split_word
        local_max = len(text_word)
        codes = tokenizer.encode_plus(text_word,
                                      return_offsets_mapping=True,
                                      is_split_into_words=True,
                                      max_length=local_max,
                                      truncation=True,
                                      return_length=True,
                                      padding="max_length")
        input_ids = torch.tensor(codes["input_ids"]).long()
        attention_mask = torch.tensor(codes["attention_mask"]).long()
        token_type_ids = torch.tensor(codes["token_type_ids"]).long()

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_token_type_id.append(token_type_ids)

        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attention_mask = torch.stack(batch_attention_mask, dim=0).bool()
        batch_token_type_id = torch.stack(batch_token_type_id, dim=0)

        tag_seqs = bert_model(batch_input_ids,
                         batch_token_type_id,
                         batch_attention_mask)

        tag_seqs = tag_seqs * batch_attention_mask
        tag_seqs = tag_seqs.numpy()

        event_res = []
        for i, tag_seq_list in enumerate(tag_seqs):
            # print(tag_seq_list)
            tag_seq_list = [id2label.get(tag, "O") for tag in tag_seq_list]

            # print(tag_seq_list)

            # event_res = {}

            extract_res = extract_entity(tag_seq_list)
            # extract_res = [ for ex in extract_res if ]
            # extract_res_n = []
            # for e_res_s, e_res_e, e_res_i in extract_res:
            #     if offset[e_res_e]-offset[e_res_s] == e_res_e-e_res_s:
            #         extract_res_n.append((e_res_s, e_res_e, e_res_i))
            #     else:



            span_res = []
            span_res_cache = []
            sub_iv = 0
            for cut in cut_point:
                while sub_iv < len(extract_res):
                    e_res = extract_res[sub_iv]
                    if e_res[1]-1<cut:
                        span_res_cache.append(e_res)
                        sub_iv += 1
                    elif e_res[0]-1>cut:
                        if span_res_cache:
                            span_res.append(span_res_cache)
                            span_res_cache = []
                        break
                    else:
                        span_res.append(span_res_cache)
                        span_res_cache = [e_res]
                        sub_iv += 1
                        break
            if span_res_cache:
                span_res.append(span_res_cache)
            for sub_span in span_res:
                event = dict()
                for e_res in sub_span:
                    key = id2role[int(e_res[2])]
                    span = input_text_[e_res[0] - 1:e_res[1] - 1]

                    p_off = e_res[1]-e_res[0]
                    if e_res[0] == 0:
                        continue
                    r_off = offset[e_res[1] - 2]-offset[e_res[0] - 1]+1

                    if p_off != r_off:
                        r_start = offset[e_res[0] - 1]
                        r_end = offset[e_res[1] - 2]+1
                        # print(r_start, r_end)
                        span = input_text[r_start:r_end]
                        # print(span_)


                    event.setdefault(key, [])
                    if span not in event[key]:
                        event[key].append(span)
                        event[key].sort()


                tempt_res.append(event)
    # tempt_res = merge_event(tempt_res)

    # for event in tempt_res:
    #     print(event)
    #
    #
    # print(tempt_res)
    final_res = []
    for event in tempt_res:
        if len(event) < 2:
            continue
        # print(event)
        final_res.append(event)

    # final_res = merge_event(tempt_res)

    return final_res


if __name__ == "__main__":
    with open("badcase.json", "r") as f:
        datas = f.read()
    datas = json.loads(datas)
    for data in datas[2:]:
        # print(data)
        # data = json.loads(data)
        # print(data["event"])
        content = data["text"]
        print(data["predict"])


        # if len(content) < 500:
        #
        extractor_event = bert2extract_ner(content)

        print(extractor_event)
        print("="*200)
        for event in data["event"]:
            print(event)
        break
