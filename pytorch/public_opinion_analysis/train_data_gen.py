#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import hanlp
from functools import reduce
from nlp_applications.data_loader import load_json_line_data
train_path = "D:\data\篇章级事件抽取\\duee_fin_train.json\\duee_fin_train.json"
dev_path = "D:\data\篇章级事件抽取\\duee_fin_dev.json\\duee_fin_dev.json"

train_data = load_json_line_data(train_path)
dev_data = load_json_line_data(dev_path)

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
# print(len(train_data))
train_data_list = []
for iv, data in enumerate(train_data):
    # for event in data.get("event_list", []):
    #     print(event["event_type"])
    # if iv < 2050:
    #     continue
    # if iv < 1843:
    #     continue
    if iv == 500:
        break

    content = data["text"]

    sentences_list = [data["title"]] + content.split("\n")

    res = HanLP(sentences_list)

    # print(content)
    # print(res["tok/fine"])
    # print(res["ner/ontonotes"])
    data_item = []

    print("idx {} {}".format(iv, data["id"]))
    for i, sentence in enumerate(sentences_list):
        d = dict()
        tokens = res["tok/fine"][i]
        k = 0
        print(i)
        print(sentence)
        print([sentence])
        print(tokens)
        # if data["id"] == "e55ffdd8af2e77df58b3020c8ee5502a" and i == 2:
        #     tokens[2] = "2014 年"
        # elif data["id"] == "e55ffdd8af2e77df58b3020c8ee5502a" and i == 6:
        #     tokens[4] = "2020 年"
        # elif data["id"] == "2e88e28052fcabe82c5b7f448f87cd17" and i == 2:
        #     tokens[8] = "SZ 000639"
        # elif data["id"] == "6b4c714c9a3d9a4e9394f589966f6cf5" and i == 7:
        #     tokens[7] = "2018 年"
        # elif data["id"] == "262eeefb326edc1029e14bb2d13402a8" and i == 1:
        #     tokens[18] = "股权 标"
        # elif data["id"] == "6236bc1f2cbd34fa545e25d95edb7010" and i == 2:
        #     tokens[2] = "王 嫣"
        # elif data["id"] == "3d33854e9e046b0b668b67063ab77340" and i == 4:
        #     tokens[49] = "满 足"
        #     tokens[109] = "预 算"
        # elif data["id"] == "8bcb5364f4b34499849c98264b440483" and i == 7:
        #     tokens[60] = "2030 年"
        # elif data["id"] == "c4a092e167e5c50bffb398c538804f71" and i == 2:
        #     tokens[4] = "20 日"
        # elif data["id"] == "41eaf44dddad0885ab62ca96afb005f6" and i == 6:
        #     tokens[8] = "Scott Sheffield"
        # elif data["id"] == "5609bb3c799efc2e0fc64f98dd4cdcf2" and i == 5:
        #     tokens[88] = "3700 万"
        # elif data["id"] == "5a9d3db5b82340c26536112268641200" and i == 1:
        #     tokens[1] = "8 月"
        # elif data["id"] == "4b99e17c7b90f2c22dc42c2bcde0f251" and i == 3:
        #     tokens[29] = "WeLab Holdings Limited"
        # elif data["id"] == "83f935da0122a771c124c5f554501130" and i == 0:
        #     tokens[31] = "公       告"
        # elif data["id"] == "83f935da0122a771c124c5f554501130" and i == 3:
        #     tokens[9] = "公       告"
        # elif data["id"] == "9f0c4c57e9c90b8091c55c49a2b0ffac" and i == 4:
        #     tokens[5] = "王永\u3000\u3000金"
        # elif data["id"] == "33638f45ad5156b9686531885d4d6e1a" and i == 0:
        #     tokens[52] = "T02 5G"
        # elif data["id"] == "33638f45ad5156b9686531885d4d6e1a" and i == 2:
        #     tokens[75] = "Tier 1"
        # elif data["id"] == "fbe032ab57538a286e47a07c6ede52a0" and i == 0:
        #     tokens[8] = "Blue Canoe"
        # elif data["id"] == "aa7ca52e077310f4eded9649a8961036" and i == 4:
        #     tokens[29] = "2019 年"
        #     tokens[34] = "2634 万"
        #     tokens[45] = "2019 年"
        #     tokens[46] = "1.54 万"
        # elif data["id"] == "a27e21e9c4939bf77a2845cb53e3f3da" and i == 33:
        #     tokens[8] = "2019 年"

        for j, token in enumerate(tokens):
            # token_len = len(token)
            # while sentence[k:k+token_len] != token:
            #     k += 1
            # d[j] = (k, k+token_len)

            token_len = 1
            if data["id"] == "9f0c4c57e9c90b8091c55c49a2b0ffac":
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "bda1fa9f84f4f6c92e7fd54ebce9b40a" and i == 1:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "7caf3963190da8bf4306f1acc01e7a62" and i == 3:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "666804dc086cbdf9d8a1d9c20c00c168" and i == 2:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "666804dc086cbdf9d8a1d9c20c00c168" and i == 31:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1

            elif data["id"] == "666804dc086cbdf9d8a1d9c20c00c168" and i == 33:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "afcaf9e25aeb214fc576e6aad86a9379" and i in [2, 3]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2003", "") != token:
                    token_len += 1
            elif data["id"] == "8d9061c98dba6713b632191044438513" and i in [3, 4, 5, 6, 7]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2003", "") != token:
                    token_len += 1
            elif data["id"] == "158cc1d0919c836ce538407f270a0de8" and i in [9]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "8c7842299c4a1bb36b7f2766b857b79b" and i in [3]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "a189fb873c16acaf2fd8d9eaf45c0861" and i in [1]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2003", "") != token:
                    token_len += 1
            elif data["id"] == "f4b0eb788877b76c6f335e346fd799b8" and i in [8, 9, 10, 11, 12]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "125c08280669abfd42696d8ea5f186ed" and i in [8]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "5797b2897601010fb66a5d61231530d8" and i in [4, 9]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "bc80aa7f20863891e8c3675cd380cba9" and i in [1]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "80261d33723cc3282dfe213b2bd5ef4d" and i in [2]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "2e053716e26319a940ff97c95b22f84a" and i in [2,3,4,5,6,7,8]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2002", "") != token:
                    token_len += 1
            elif data["id"] == "a8562588519e6650641a474a4a5bbbbf" and i in [8, 9, 10, 11]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "38aa3624f0d7e75a30ce457c6480c25b" and i in [1]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "26111146271e23cc7094dfe7b47c9e1c" and i in [3]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "3a57e3582154091e7b423cd803bf21e8" and i in [4, 6]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2003", "") != token:
                    token_len += 1
            elif data["id"] == "5eba572a15be1673738c5d32f8da4881" and i in [2]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "474fff38a341d43cc3a42f202f324919" and i in [2, 3]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2003", "") != token:
                    token_len += 1

            elif data["id"] == "4405e71dbb9d7c2f50038662b345958d" and i in [3]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1

            elif data["id"] == "f0e9aa585b34ae573c0ab659b1bc0f0d" and i in [1]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            elif data["id"] == "bb58f1e88ec5ba2d0be1a3c270f0f768" and i in [2, 3, 4, 5, 6]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2002", "") != token:
                    token_len += 1
            elif data["id"] == "96fc6aefbd535b06f90604428dbc8804" and i in [2, 4, 5, 6, 7, 8, 9]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u2003", "") != token:
                    token_len += 1
            elif data["id"] == "d50a1d9a58202c21432e1881b923f069" and i in [0]:
                while sentence[k:k + token_len].replace(" ", "").replace("\u3000", "") != token:
                    token_len += 1
            else:
                while sentence[k:k + token_len].replace(" ", "") != token:
                    token_len += 1
            d[j] = (k, k + token_len)
            k += token_len




        label = []
        for item in res["ner/ontonotes"][i]:
            if item[1] not in ["ORG", "PERSON"]:
                continue
            item_range = reduce(lambda x,y: (x[0], y[1]), [d[it] for it in range(item[2], item[3])])
            label.append({
                "start": item_range[0],
                "end": item_range[1],
                "span": item[0],
                "type": item[1]
                         })
        # print(label)
        data_item.append({
            "sentence": sentence,
            "label": label
        })
        # for j, s in enumerate(sentences):

    train_data_list.append({
        "data_item": data_item,
        "data": data
    })
import json
data_json = json.dumps(train_data_list)

with open("train.json", "w") as f:
    f.write(data_json)


