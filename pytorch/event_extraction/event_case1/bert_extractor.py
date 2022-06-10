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
from pytorch.event_extraction.event_case1.train_data_v2 import rt_data
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

tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext")
bert_model = EventModelV1(config)
model_pt = torch.load("all-finance.pt")
bert_model.load_state_dict(model_pt)


def merge_event(event_list):
    # n_list = []
    while True:
        n_list = []
        for event in event_list:
            if n_list:
                last_event = n_list.pop()
                if len(event) == 2 and "领投方" in event and "投资方" in event:
                    for k, v in last_event.items():
                        event[k] = v
                elif len(event) == 1 and "投资方" in event:
                    for k, v in last_event.items():
                        event[k] = v
                else:
                    cstate = 0
                    for k, v in last_event.items():
                        if k not in event or v != event[k]:
                            cstate = 1
                            break
                    sstate = 0
                    for k, v in event.items():
                        if k not in last_event or v != last_event[k]:
                            sstate = 1
                            break
                    if cstate == 0:
                        pass
                    elif sstate == 0:
                        event = last_event
                    else:
                        n_list.append(last_event)

            n_list.append(event)
        if len(n_list) == len(event_list):
            break
        event_list = n_list

    investee = None
    for event in event_list:
        if investee is None and "被投资方" in event and len(event["被投资方"]) == 1:
            investee = event["被投资方"]

            # 如果只有投资者，而只有被投资者
        if "被投资方" not in event and investee:
            event["被投资方"] = investee

    dsu = DSU(len(event_list))
    for i, event_i in enumerate(event_list):
        for j, event_j in enumerate(event_list):
            if j <= i:
                continue
            # print(event_i.get("被投资方", "a"), event_j.get("被投资方", "b"), "==========")
            # print(event_i.get("融资轮次", "a"), event_j.get("融资轮次", "b"), "==========")
            if event_i.get("被投资方", "a") == event_j.get("被投资方", "b") and event_i.get("融资轮次", "a") == event_j.get("融资轮次", "b"):
                # print(i, j, "==========")
                dsu.union(i, j)

    cluster_list = []
    for i in range(len(event_list)):
        if len(cluster_list) == 0:
            cluster_list.append([i])
        else:
            state = 1
            for j_cluster in cluster_list:
                if dsu.find(i) == dsu.find(j_cluster[0]):
                    j_cluster.append(i)
                    state = 0
                    break
            if state:
                cluster_list.append([i])
    print(cluster_list)
    final_list_middle = []
    for cluster in cluster_list:
        event = {}
        for idx in cluster:
            for k, v in event_list[idx].items():
                event[k] = list(set(v+event.get(k, [])))
        final_list_middle.append(event)
    final_list = []
    # investee = None
    for i, event1 in enumerate(final_list_middle):
        state = 1
        # 移除规则 {'投资方': ['沃衍资本'], '被投资方': ['帝奥微']} 没有其他信息要移除
        if len(event1) == 2 and "投资方" in event1 and "被投资方" in event1:
            continue
        # 移除规则 {'事件时间': ['5月份'], '融资金额': ['604.92亿元']}
        if len(event1) == 2 and "事件时间" in event1 and "融资金额" in event1:
            continue

        for event2 in final_list_middle[i+1:]:
            if event1 == event2:
                state = 0
                break
        if state:
            if "领投方" in event1:
                event1.setdefault("投资方", [])
                for item in event1["领投方"]:
                    if item not in event1["投资方"]:
                        event1["投资方"].append(item)
            if "投资方" in event1:
                event1["投资方"].sort()
            if "领投方" in event1:
                event1["领投方"].sort()
            for k, v in event1.items():
                if k not in ["领投方", "投资方"]:
                    v.sort(key=lambda x: len(x))
                    event1[k] = v[-1:]
            final_list.append(event1)

    return final_list


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

    # 简称
    pattern_list = ["（简称“([\u4e00-\u9fa5]+?)”）",
                    "（以下简称：([\u4e00-\u9fa5]+?)）",
                    "（以下简称“([\u4e00-\u9fa5]+?)”）",
                    "(以下简称([\u4e00-\u9fa5]+?))",
                    "(简称：([\u4e00-\u9fa5]+?))"]
    print("简称")
    for pattern in pattern_list:
        for res in re.findall(pattern, input_text):
            print(res)

    # 文章时间
    pattern = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}"
    datetimes = re.findall(pattern, input_text)
    print("时间")
    print(datetimes)


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

            rs = re.finditer("融资", input_text_)
            for r in rs:
                print(r.span(), r.group())

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

                    if key == "被投资方" and len(span) < 2:
                        continue

                    event.setdefault(key, [])
                    if span not in event[key]:
                        event[key].append(span)

                if len(event.get("融资轮次", [])) == 2 and len(event.get("融资金额", [])) == 2:

                    pass


                tempt_res.append(event)
    tempt_res = merge_event(tempt_res)

    # for event in tempt_res:
    #     print(event)
    #
    #
    # print(tempt_res)
    final_res = []
    for event in tempt_res:
        if len(event) < 2:
            continue
        print(event)
        final_res.append(event)

    # final_res = merge_event(tempt_res)

    return final_res


if __name__ == "__main__":

    with open("finance_add.json", "r") as f:
        datas = f.read()

    datas = json.loads(datas)
    datas = pd.read_csv("badcase.csv")
    #
    for did, data in datas.iterrows():
        content = data["concent"]
        print(content)


        # if len(content) < 500:
        #
        extractor_event = bert2extract_ner(content)
            # break

    # hit_num = 0
    # rel_num = 0
    # pre_num = 0
    # bad_case = []
    # role_indicate = dict()
    # account_role = {"num": 0}
    # i = 0
    # for data in datas:
    #     if len(data["text"]) > 500:
    #         continue
    #     print(data["text"])
    #
    #     extractor_event = bert2extract_ner(data["text"])
    #     i += 1
    #
    #     if i == 3:
    #         break
