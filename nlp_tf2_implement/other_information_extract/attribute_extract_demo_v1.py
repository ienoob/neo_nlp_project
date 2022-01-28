#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import docx
import json


data_path = "D:\data\contract\合同data"

def read_json(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    return json.loads(data)
file_list = []
def f(input_path):
    if os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            n_file_path = os.path.join(input_path, file_name)
            f(n_file_path)
    else:
        file_list.append(input_path)

f(data_path)

d = dict()
for file in file_list:
    dir_name = os.path.dirname(file)
    d.setdefault(dir_name, {"doc": None, "res": None})
    if file.split(".")[-1] == "docx":
        d[dir_name]["doc"] = file
    else:
        d[dir_name]["res"] = file

# print(d)

def f2(key, key2, input_content_list):
    res = []
    for i, sentence in enumerate(input_content_list):
        if sentence.find(key) == -1:
            continue
        # print(sentence, key2)
        res.append((i, sentence.find(key2), key2))
    return res

sentence_list = []
test_sentence_list = []
dev_sentence_list = []
label_list = []

def f1(extract_key, input_res_json, input_content_list):
    extract_res_list = []
    if extract_key == "owner_subject":
        owner_list = input_res_json.get(extract_key)
        for owe in owner_list:
            content_value = owe["content"]
            if content_value.strip() == "":
                continue
            if content_value == "指天空科技股份有限公司。":
                content_value = "天空科技股份有限公司"
            elif content_value == "天空科技股份有限公司(以下称为“买方”)":
                content_value = "天空科技股份有限公司"
            extract_res = f2(owe["position"], content_value, input_content_list)
            extract_res_list.append(extract_res)
        return extract_res_list
    elif extract_key == "other_subject":
        owner_list = input_res_json.get(extract_key)
        for owe in owner_list:
            print(owe)
            content_value = owe["content"]
            if content_value.strip() == "":
                continue
            if content_value == "指在协议书中约定，被发包人/发包人省分公司接受的具有工程施工承包主体资格的当事人以及继受该当事人权利义务的合法主体。":
                continue
            if content_value == "指地圆科技股份有限公司。":
                content_value = "地圆科技股份有限公司"
            extract_res = f2(owe["position"], content_value, input_content_list)
            extract_res_list.append(extract_res)

        return extract_res_list

    elif extract_key == "sign_date":
        target = input_res_json.get(extract_key)
        if target:
            extract_res = f2(target, target, input_content_list)
            extract_res_list.append(extract_res)
    elif extract_key == "contract_name":
        target = input_res_json.get(extract_key, dict())
        if target:
            extract_res = f2(target["content"], target["content"], input_content_list)
            extract_res_list.append(extract_res)
    elif extract_key == "contract_number":
        print(input_res_json.get(extract_key))
    elif extract_key == "contract_amount":
        target = input_res_json.get(extract_key)
        print(target)
        if target[1] == 2760400.0:
            target[1] = "2760400.00"
        elif target[1] == 3978962.11:
            target[1] = "3978962.11"
        elif target[1] == 200000.0:
            target[1] = "200,000"
        elif target[1] == 4348120.0:
            target[1] = "4,348,120.00"
        elif target[1] == 799359.31:
            target[1] = "799,359.31"
        elif target[1] == 1943298.0:
            target[1] = "1,943,298.00"

        if target[0]:
            extract_res = f2(target[0], target[0], input_content_list)
            extract_res_list.append(extract_res)
        if target[1]:
            extract_res = f2(target[1], target[1], input_content_list)
            extract_res_list.append(extract_res)

    return extract_res_list

for key, itme in d.items():
    d = dict()
    document = docx.Document(itme["doc"])
    content_list = []
    print(key)
    for p in document.paragraphs:
        text = p.text.replace("\n", "").replace(" ", "")
        if text.strip():
            text = "^" + text.strip() + "$"
            content_list.append(text)
    # print(content_list)

    if itme["res"]:
        res_json = read_json(itme["res"])
        extract_res_list = f1("owner_subject", res_json, content_list)
        print(extract_res_list)
        # owner_list = res_json.get("owner_subject")
        # for owe in owner_list:
        #     print(owe)
        #     content_value = owe["content"]
        #     if content_value.strip() == "":
        #         continue
        #     if content_value == "指天空科技股份有限公司。":
        #         content_value = "天空科技股份有限公司"
        #     elif content_value == "天空科技股份有限公司(以下称为“买方”)":
        #         content_value = "天空科技股份有限公司"
        #     extract_res = f2(owe["position"], content_value, content_list)
        #     print(extract_res)
        for extract_res in extract_res_list:
            for a, b, c in extract_res:
                d[(a, b)] = "B-E"
                for bi in range(b+1, b+len(c)):
                    d[(a, bi)] = "I-E"
        for i, content in enumerate(content_list):
            sentence_list.append(content)
            content_label = [d.get((i, j), "O") for j, char in enumerate(content)]
            label_list.append(content_label)
        dev_sentence_list.append(content_list)
    else:
        test_sentence_list.append(content_list)


from nlp_applications.ner.crf_model import sent2labels, CRFNerModel
from nlp_applications.ner.evaluation import extract_entity


def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word_lower()': word.lower(),
        "is_digits": word.isdigit()
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            "-1:is_digits": word1.isdigit()
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            "+1:is_digits": word1.isdigit()
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

X_train = [sent2features(s) for s in sentence_list]
y_train = [sent2labels(s) for s in label_list]
crf_mode = CRFNerModel()
crf_mode.fit(X_train, y_train)

for ti, test_sentence_in in enumerate(test_sentence_list):
    X_test = [sent2features(s) for s in test_sentence_in]

    predict_labels = crf_mode.predict_list(X_test)

    for ii, p_label in enumerate(predict_labels):

        true_entity = extract_entity(p_label)
        # print(test_sentence_list[1][ii])

        for s, e, _ in true_entity:
            print("contract num {} sentence num {}".format(ti, ii), test_sentence_in[ii])
            print("contract num {} info".format(ti), test_sentence_in[ii][s:e])


schema_list = [
    {
        "entity_name": "owner_subject",
        "entity_cn_name": "甲方"
    },
    {
        "entity_name": "contract_name",
        "entity_cn_name": "合同名称"
    },
    {
        "entity_name": "other_subject",
        "entity_cn_name": "乙方"
    },
    {
        "entity_name": "sign_date",
        "entity_cn_name": "合同日期"
    },
    {
        "entity_name": "contract_number",
        "entity_cn_name": "合同总金额"
    },
    {
        "entity_name": "contract_currency",
        "entity_cn_name": ""
    },
    {
        "entity_name": "contract_amount",
        "entity_cn_name": ""
    },
    {
        "entity_name": "forbidden_words",
        "entity_cn_name": ""
    }
]
event_list = [
    {
        "event_type": "付款",
        "event_arg": [
            {"role": "付款阶段"},
            {"role": "付款金额大写"},
            {"role": "付款金额小写"},
            {"role": "付款比例"},
            {"role": "付款条件"}
        ]
    }
]
