#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import hanlp
import pandas as pd
from entity_describe_data_p2 import generate_label_data, generater_label_single_file
"""
    概念解释模型规则生成
"""
# for item in generater_label_single_file(25):
#     print(item)
# for item in generate_label_data():
#     print(item)
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
filter_p = {"是"}
role_list = ["ARG0", "PRED", "ARG1"]
i = 0
condition = dict()
df = []
for sentence in generate_label_data():
    print(sentence)
    i += 1

    # print(sentence)

    documents = HanLP(sentence["sentence"])
    input_document_srl = documents["srl"]


    for role_part in input_document_srl:
        spo = {}
        if len(role_part) < 3:
            continue

        for role_mention, role_key, start, end in role_part:
            if role_key not in role_list:
                continue
            if role_key == "PRED":
                spo["p"] = (role_mention, start, end)
            elif role_key == "ARG0":
                spo["s"] = (role_mention, start, end)
            elif role_key == "ARG1":
                spo["o"] = (role_mention, start, end)

        if "p" not in spo:
            continue
        if "o" not in spo:
            continue
        if "s" not in spo:
            continue

        # print(spo)
        # print(documents["dep"])
        sentence_dep = documents["dep"]
        sentence_pos = documents["pos/pku"]
        sentence_pos_word = documents["tok/fine"]
        sub = spo["s"]
        obj = spo["o"]
        pre = spo["p"]
        sub_dep_full = "-".join([sentence_dep[iv][1] for iv in range(sub[1], sub[2])])
        sub_pos_full = "-".join([sentence_pos[iv] for iv in range(sub[1], sub[2])])
        obj_pos_full = "-".join([sentence_pos[iv] for iv in range(obj[1], obj[2])])
        obj_dep_full = "-".join([sentence_dep[iv][1] for iv in range(obj[1], obj[2])])
        # 条件
        key = (sub[2]-sub[1],  # 主语词语数量
               sentence_pos[sub[2]-1],  # 主语词性
               sentence_pos[obj[2]-1],  # 宾语词性
               sentence_dep[pre[1]][1],  # 谓语句法
               sentence_dep[sub[2]-1][1], # 主语句法
               sentence_dep[obj[2]-1][1], # 宾语词性
               pre[1]-sub[2],
               sub[2]-1,
               spo["p"][0],
               sub_dep_full,
               sub_pos_full)
        df.append({"s": spo["s"][0], "p": spo["p"][0], "o": spo["o"][0],
                   "sub_word_num": sub[2]-sub[1],
                   "sub_last_pos": sentence_pos[sub[2]-1],
                   "obj_last_pos": sentence_pos[obj[2]-1],
                   "pre_dep": sentence_dep[pre[1]][1],
                   "sub_last_dep": sentence_dep[sub[2]-1][1],
                   "obj_last_dep": sentence_dep[obj[2]-1][1],
                   "pre_sub_dis": pre[1]-sub[2],
                   "sub_loc": sub[2]-1,
                   "sub_pos_full": sub_pos_full,
                   "sub_dep_full": sub_dep_full,
                   "obj_pos_full": obj_pos_full,
                   "obj_dep_full": obj_dep_full
                   })
        condition.setdefault(key, 0)
        condition[key] += 1


condition_sort = [(k, v) for k, v in condition.items()]
condition_sort.sort(key=lambda x: x[1], reverse=True)

# for i in range(100):
#     print(condition_sort[i])

df = pd.DataFrame(df)

print(df.shape)
df.to_csv("data.csv", index=False)

"""
    import pandas as pd

    df = pd.read_csv("data.csv")
    
    # print(df.head(5))
    
    new_data = df[(df["sub_word_num"] == 4) & (df["sub_last_pos"] == "q") & (df["obj_last_pos"] == "w")]
    print(new_data[["s", "p", "o"]].head(5))
    
    new_data = df[(df["sub_word_num"] == 4) & (df["sub_last_pos"] == "q") & (df["obj_last_pos"] == "q")]
    print(new_data[["s", "p", "o"]].head(5))
    
    df_group = df.groupby(["sub_word_num", "sub_pos_full", "sub_dep_full", "pre_dep", "obj_pos_full", "obj_dep_full", "pre_sub_dis", "p"])
    item_list = []
    for key, item in df_group:
        item_list.append((key, len(item)))
    print(len(item_list))
    item_list.sort(key=lambda x: x[1], reverse=True)
    print(" ".join(["sub_word_num", "sub_pos_full", "sub_dep_full", "pre_dep", "obj_pos_full", "obj_dep_full", "pre_sub_dis", "p"]))
    for i in range(20):
        print(item_list[i])
"""
