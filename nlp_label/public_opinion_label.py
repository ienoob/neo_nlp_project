#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/24 14:45
    @Author  : jack.li
    @Site    : 
    @File    : public_opinion_label.py

"""
import os
import json
from nlp_tf2_implement.ner.evaluation import extract_entity
path = "D:\\xxxx\\opinion_analysis_v1.json"




def get_label():
    label_dict = dict()
    path = "D:\\xxxx\\train_data\\"
    file_list = os.listdir(path)
    file_list.sort()
    i = 0
    type_set = set()
    for file in file_list:
        if file.endswith("bio"):
            i += 1
            # print(file, i)
            item_id = file.split(".")[0]
            file_path = path + file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
            except Exception as e:
                try:
                    with open(file_path, "r", encoding="GB2312") as f:
                        data = f.read()
                except Exception as e:
                    with open(file_path, "r", encoding="gbk") as f:
                        data = f.read()
            data_list = []
            cache_list = []
            for row in data.split("\n"):
                if row != "":
                    row_split = row.split(" ")
                    if len(row_split) == 2:
                        cache_list.append(row_split)
                    elif len(row_split) == 3:
                        cache_list.append((" ", row_split[-1]))
                    else:
                        raise Exception
                else:
                    data_list.append(cache_list)
                    cache_list = []
            for item in data_list:
                # print(item)
                item_sentence = [it[0] for it in item]
                item_label = [it[1] for it in item]
                # print(item_label)
                # print(item_sentence)
                entitys = extract_entity(item_label)
                entity = [item_sentence[et[0]:et[1]] for et in entitys]
                entity_type = [et[2] for et in entitys]
                # print(entity_type)
                for e_type in entity_type:
                    type_set.add(e_type)
            # print(len(data_list))
            # print(len(train_dict[item_id]))
            # assert len(data_list) == len(train_dict[item_id])+1

            label_dict[item_id] = data_list

    # print(i)
    assert len(type_set) == 2

    return label_dict

with open(path, "r") as f:
    data = f.read()

d = {
    "O": "O",
    "ORG-S": "B-ORG",
    "ORG-I": "I-ORG",
    "PERSON-S": "B-PERSON",
    "PERSON-I": "I-PERSON",
}

data_dict = json.loads(data)
# target = 2
label_dict = get_label()

import copy
def split_512(input_list):
    n_split_list = []
    n_split_num = []
    cache_list = []
    cache_num = []
    cache_len = 0
    for i, sentence in enumerate(input_list):
        # print(sentence)
        if cache_len+len(sentence) <= 510:
            cache_list.append(sentence)
            cache_len += len(sentence)
            cache_num.append(i)
            # print(cache_num)
        else:
            # print(cache_list)
            n_split_list.append(copy.deepcopy(cache_list))
            n_split_num.append(copy.deepcopy(cache_num))
            cache_list = [sentence]
            cache_len = len(sentence)
            cache_num = [i]
        # print(n_split_list)
    if cache_len:
        n_split_list.append(cache_list)
        n_split_num.append(cache_num)

    return n_split_list, n_split_num
import hashlib
import pandas as pd
old_path = "D:\\xxxx\\opinion_analysis_v3.csv"
old_df = pd.read_csv(old_path)
old_opinion_label = dict()
for idx, row in old_df.iterrows():

    text_id = hashlib.md5(row["text"].encode()).hexdigest()
    old_opinion_label[(row["id"], text_id, row["entity_list"])] = row["label"]





train_dict = dict()
train_nw_list = []
# import pandas as pd
for ii, item in enumerate(data_dict):
#     # if ii != target:
#     #     continue
#     # print(item)
#     print(item["other"]["title"])
#     print(item["text"])
#     for i, sub in enumerate(item["batch_label"]):
#         sentence = item["batch_sentence"][i]
#         sub = [d[x] for x in sub]
#         res = ["".join(sentence[rs[0]:rs[1]]) for rs in extract_entity(sub)]
        # print(res)
    item_label_list = []
    item_sentence_list = []
    for itm in label_dict[item["id"]]:
        # print(item)
        item_sentence = [it[0] for it in itm]
        item_label = [it[1] for it in itm]
        item_label_list.append(item_label)
        item_sentence_list.append(item_sentence)
    if item["id"] in ["f805ed711169aa33185db13dbb07081b", "4a6617e4d1077ba3f6d509aa237741b3", "27dfb00137c2920abf10433ab97a44b2",
                      "5a62798a3fab8ba423fdea64c18f9771", "7caf3963190da8bf4306f1acc01e7a62"]:
        continue

    # print(item_sentence_list)
    print(item["id"])
    event_label_res = item["data_event"]
    split_list, split_num = split_512(item_sentence_list)
    # print(split_list)
    for iii, sl in enumerate(split_list):
        sl_len = sum([len(x) for x in sl])
        print(sl_len)
        assert sl_len < 512

        sub_split_num = split_num[iii]
        sub_entity_list = []
        sub_entity_full_list = []
        sub_entity_bio_label = []
        for sub_i in sub_split_num:
            # print(sub_label)
            sub_label = item_label_list[sub_i]
            assert len(sub_label) == len(item_sentence_list[sub_i])
            sub_entity = extract_entity(sub_label)
            sub_entity_bio_label.append(sub_label)
            # sub_entity_list = []
            for start, end , tp in sub_entity:
                sss_entity = "".join(item_sentence_list[sub_i][start:end])
                if sss_entity not in sub_entity_list:
                    sub_entity_list.append(sss_entity)
                sub_entity_full_list.append((sss_entity, start, end))
            # print(sub_entity_list)

        event_set = set()
        for iid in sub_split_num:
            if str(iid) in event_label_res:
                for e_label in event_label_res[str(iid)]:
                    event_set.add(e_label)
        sub_num_str = "$".join([str(iid) for iid in sub_split_num])
        sub_text = "\n".join(["".join(sentence) for sentence in sl])
        sub_entity_pn_dict = dict()
        for s_entity in sub_entity_list:
            pre_label = -1
            if (item["id"], hashlib.md5(sub_text.encode()).hexdigest(), s_entity) in old_opinion_label:
                pre_label = old_opinion_label[(item["id"], hashlib.md5(sub_text.encode()).hexdigest(), s_entity)]
            else:
                raise Exception
            sub_entity_pn_dict[s_entity] = pre_label
        sub_entity_pn = []
        for entity, start, end in sub_entity_full_list:
            sub_entity_pn.append({"entity": entity, "start": start, "end": end, "pn_label": sub_entity_pn_dict[entity]})

        assert len(sl) == len(sub_entity_bio_label)
        for iiii, sentence in enumerate(sl):
            assert len(sentence) == len(sub_entity_bio_label[iiii])
        train_nw_list.append({"id": item["id"], "sub_num_str": sub_num_str, "text": sub_text, "entity_pn": sub_entity_pn,
                              "sentence_cut": sl, "entity_bio": sub_entity_bio_label, "event_label": list(event_set)})


    # print(item)
    # new_data = {
    #     "id": item["id"],
    #     "batch_sentence": item_sentence_list[:-1],
    #     "batch_label": item_label_list[:-1],
    #     "data_event": item["data_event"]
    # }
    # train_nw_list.append(new_data)

json_dumps = json.dumps(train_nw_list)
with open("D:\\xxxx\\opinion_analysis_v4.json", "w") as f:
    f.write(json_dumps)

# df = pd.DataFrame(train_nw_list)
# df.to_csv("D:\\xxxx\\opinion_analysis_v4.csv", index=False)

print("end============")
# new_path = "D:\\xxxx\\opinion_analysis_v1.json"
    # train_dict[item["id"]] = item["batch_label"]
#     print(ii, "=================================")
#
#     text = item["other"]["title"] + "\n" + item["text"]
#     with open("D:\\xxxx\\train_data\\{}.txt".format(item["id"]), "w", encoding="utf-8") as f:
#         f.write(text)
#
#     # print(item["batch_label"])
#
#     # break
# print("\033[0;31;40m我是小杨我就这样\033[0m")

