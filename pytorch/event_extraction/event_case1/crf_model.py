#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json
from nlp_applications.ner.crf_model import CRFNerModel
from pytorch.event_extraction.crf_utils import DSU

trigger_set = {'集资', '募集', '补助', '政府补助', '扶持', '融资', '发行股份', '非公开发行', '投资', '发行债券', '融资券', '增发', '挂牌上市', '筹得', '筹集', '非公开发行股票', '发行', '募资', '募集资金'}



def split(input_str):
    split_char = {"。", "\n", "\r"}
    not_add_char = {"\r", "\n"}
    start = 0
    for i, i_char in enumerate(input_str):
        if i_char not in split_char:
            continue
        if i > start:
            if i_char in not_add_char:
                sub_str = input_str[start:i]
            else:
                sub_str = input_str[start:i+1]
            sub_str = sub_str.strip()
            yield sub_str
        start = i+1
    if start<len(input_str):
        sub_str = input_str[start:]
        yield sub_str


# def train():
#    from pytorch.event_extraction.event_case1.train_data_v2 import crf_train_list
#
#    return crf_train_list
from pytorch.event_extraction.event_case1.train_data_v2 import rt_data


train_data_lists = rt_data["crf_data"]["train"]
dev_data_lists = rt_data["crf_data"]["dev"]


role2id = rt_data["role2id"]
id2role = {v:k for k, v in role2id.items()}


def train_crf_model():

    from nlp_applications.ner.evaluation import metrix_v2, extract_entity, eval_metrix

    def word2features(sent, i):
        word = sent[i]

        features = {
            'bias': 1.0,
            'word_lower()': word.lower(),
            "word_isdigit()": word.isdigit()
        }
        if i > 0:
            word1 = sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.isdigit()': word1.isdigit()
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.isdigit()': word1.isdigit()
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

    def sent2labels(sent):
        return [label for label in sent]

    # for train in train_data_lists:
    #     print(train["text"])
    #     print(train["label"])

    crf_mode = CRFNerModel()
    crf_mode.save_model = "finance.model"
    X_train = [sent2features(s["text"]) for s in train_data_lists]
    # print(X_train[0])
    y_train = [sent2labels(s["label"]) for s in train_data_lists]

    X_dev = [sent2features(s["text"]) for s in dev_data_lists]
    # print(X_train[0])
    y_dev = [sent2labels(s["label"]) for s in dev_data_lists]
    # print(y_train[0])

    crf_mode.fit(X_train, y_train)

    predict_labels = crf_mode.predict_list(X_dev)

    role_indicate = dict()
    for i, pred in enumerate(predict_labels):
        real_row = y_dev[i]

        true_entity = extract_entity(real_row)
        pred_entity = extract_entity(pred)

        for e in true_entity:
            role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
            role_indicate[e[2]]["real"] += 1

        for e in pred_entity:
            role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
            role_indicate[e[2]]["pred"] += 1

        for e in true_entity:
            if e in pred_entity:
                role_indicate[e[2]]["hit"] += 1

    for role_id, role_ind in role_indicate.items():
        print("{} : {}".format(id2role[int(role_id)], eval_metrix(role_ind["hit"], role_ind["real"], role_ind["pred"])))

    crf_mode.dump_model()


def merge_event(event_list):
    final_res = []
    event_list.sort(key=lambda x: len(x))

    for i, sub_res in enumerate(event_list):
        state = 1
        for sub_j_res in event_list[i+1:]:
            s = 1
            for k, v in sub_res.items():
                if k not in sub_j_res:
                    s = 0
                    break
                if v != sub_j_res[k]:
                    sub_v_num = 0
                    for sub_v in v:
                        if sub_v in sub_j_res[k]:
                            sub_v_num += 1
                    if sub_v_num != len(v):
                        s = 0
                        break
            if s:
                state = 0
                break
        if state:
            if "领投方" in sub_res:
                if "投资方" not in sub_res:
                    sub_res["投资方"] = sub_res["领投方"]
                else:
                    for item in sub_res["领投方"]:
                        if item not in sub_res["投资方"]:
                            sub_res["投资方"].append(item)
                        sub_res["投资方"].sort()

            final_res.append(sub_res)

    pair_list = []
    dsu = DSU(len(final_res))
    for i, event_i in enumerate(final_res):
        for j, event_j in enumerate(final_res):
            if j <= i:
                continue
            if event_i.get("被投资方", "a") == event_j.get("被投资方", "b") and event_i.get("融资轮次") == event_j.get("融资轮次"):
                pair_list.append((i, j))
                dsu.union(i, j)
            # if (event_i.get("融资轮次") is None or event_j.get("融资轮次") is None) and event_i.get("红杉中国") == event_j.get("红杉中国"):
            #     pair_list.append((i, j))
            #     dsu.union(i, j)

            if (event_i.get("被投资方") is None or event_j.get("被投资方") is None) and event_i.get("融资轮次", "a") == event_j.get("融资轮次", "b"):
                pair_list.append((i, j))
                dsu.union(i, j)

            if event_i.get("融资轮次") is None and  event_i.get("领投方", "a") == event_j.get("领投方", "b"):
                pair_list.append((i, j))
                dsu.union(i, j)

    cluster_list = []
    for i in range(len(final_res)):
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

    final_event_list = []
    for cluster in cluster_list:
        event = {}
        for idx in cluster:
            for k, v in final_res[idx].items():
                event[k] = v
        # 简单筛选
        if len(event) < 2:
            continue
        if "被投资方" not in event and "投资方" not in event:
            continue
        final_event_list.append(event)
    return final_event_list

def crf_extractor(input_text):
    model_path = "finance.model"

    model = CRFNerModel()
    model.save_model = model_path
    model.load_model()

    tempt_res = []
    # text_sentence = re.split("[。\r\n]", input_text)
    text_sentence = split(input_text)
    for i, sentence in enumerate(text_sentence):
        t_state = 0
        for t_word in trigger_set:
            if t_word in sentence:
                t_state = 1
                break
        if t_state == 0:
            continue
        event_res = {}
        extract_res = model.extract_ner(sentence)
        for e_res in extract_res:
            # event_res.setdefault(e_res[2], set())
            key = id2role[int(e_res[2])]
            event_res.setdefault(key, [])
            if e_res[3] not in event_res[key]:
                loc_span = e_res[3]
                if loc_span[-1] == ",":
                    loc_span = loc_span[:-1]
                event_res[key].append(loc_span)
            # event_res[key] = e_res[3]
        if len(event_res) < 2:
            continue
        for k, v in event_res.items():
            v.sort()
            event_res[k] = v
        # merge event
        # for res in event_res:

        tempt_res.append(event_res)

    # merge stage
    final_res = merge_event(tempt_res)


    return final_res

if __name__ == "__main__":

    print(role2id)
    train_crf_model()
