#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/6/7 20:23
    @Author  : jack.li
    @Site    : 
    @File    : public_entity_evaluation.py

"""
import os
import pandas as pd
from pytorch.knowledge_graph.entity_kg import is_not_valid
from nlp_tf2_implement.ner.evaluation import extract_entity, eval_metrix

predict_path = "D:\\xxxx\\public_opinion_company_label.csv"
path = "D:\\xxxx\\ad_train_data\\"

df = pd.read_csv(predict_path)

i = 0
predict_num = 0.0
hit_num = 0.0
true_num = 0.0
label_document_num = 0
for file in os.listdir(path):
    if file.endswith("bio"):
        label_document_num += 1
        i += 1
        # print(file, i)
        item_id = file.split(".")[0]
        print(item_id)
        file_path = path + file
        sub_item = df[df["trz_id"]==item_id]

        entity_predict_set = {row["entity"] for idx, row in sub_item.iterrows() if not is_not_valid(row["entity"])}
        predict_num += len(entity_predict_set)
        print("predict ", entity_predict_set)
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
        entity_set = set()
        for item in data_list:
            # print(item)
            item_sentence = [it[0] for it in item]
            item_label = [it[1] for it in item]
            # print(item_label)
            # print(item_sentence)
            entitys = extract_entity(item_label)
            for entity in entitys:
                if entity[2] == "COMPANY":
                    entity_set.add("".join(item_sentence[entity[0]:entity[1]]))
        true_num += len(entity_set)
        hit_num += len(entity_set & entity_predict_set)
        print(entity_set)
        print(entity_predict_set-entity_set)
            # entity_type = [et[2] for et in entitys]
            # print(entity_type)
            # for e_type in entity_type:
            #     type_set.add(e_type)
print("document num {}".format(label_document_num))
print(hit_num, true_num, predict_num)
print(eval_metrix(hit_num, true_num, predict_num))