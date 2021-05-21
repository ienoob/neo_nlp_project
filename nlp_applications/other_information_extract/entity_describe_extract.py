#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/17 23:12
    @Author  : jack.li
    @Site    : 
    @File    : entity_describe_extract.py

    任务名称：实体 实体描述分析，目的是抽取出文本中的实体以及对应的实体描述。

"""
import re
from ltp import LTP
from nlp_applications.data_loader import load_json_line_data

data_path = "D:\data\语料库\wiki_zh_2019\wiki_zh\AA\\wiki_00"

data = load_json_line_data(data_path)




input_data = "文学（），在最广泛的意义上，是任何单一的书面作品。"
pattern = "^(.+?)是(.+?)[,，。.]"

# print(re.findall(pattern, input_data))

filter_p = {"是", "为"}

sentence = "文学批评是指文学批评者对其他人作品的评论和评估，有时也会用来改进及提升文学作品。"


def single_sentence(input_sentence):
    ltp = LTP()
    seg, hidden = ltp.seg([input_sentence])
    words = seg[0]

    pos = ltp.pos(hidden)[0]
    roles = ltp.srl(hidden, keep_empty=False)[0]

    # print(words)
    # print(roles)
    role_list = ["A0", "A1", "A2", "A3", "A4"]
    # print(words)
    spo_list = []
    for role in roles:
        r_indx, r_list = role

        p_value = words[r_indx]
        r_list = list(filter(lambda x: x[0] in role_list, r_list))
        if len(r_list) != 2:
            continue
        sub = r_list[0]
        obj = r_list[1]

        if sub[0] not in role_list:
            continue
        if obj[0] not in role_list:
            continue
        if sub[2] >= r_indx:
            continue
        if obj[1] <= r_indx:
            continue
        # print(pos[sub[2]])
        # 词性过滤
        if pos[sub[2]] not in ["n"]:
            continue
        # 谓语过滤
        if p_value not in filter_p:
            continue

        sub_value = words[sub[1]:sub[2]+1]

        obj_value = words[obj[1]:obj[2]+1]
        # print("".join(sub_value), p_value, "".join(obj_value))

        spo_list.append(("".join(sub_value), p_value, "".join(obj_value)))

    return spo_list


def entity_describe_analysis(input_sentence_list):
    spo_res = []
    for i, sentence in enumerate(input_sentence_list):
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        out_spo_list = single_sentence(sentence)
        if out_spo_list:
            spo_res.append((sentence, out_spo_list))

    return spo_res
contents = "AnyShare由上海爱数信息技术股份有限公司自主研发的一款软硬件一体化产品，主要面向企业级用户，提供非结构化数据管理方案。"
sentence_list = contents.split("。")
# single_sentence(sentence)
out_spo = entity_describe_analysis(sentence_list)

for sentence, spo in out_spo:
    print(sentence)
    print(spo)
# for i, dt in enumerate(data):
#     if i >= 5:
#         break
#     print(dt["title"])
#     # print(dt["text"])
#     sentence_list = re.split("[。\n]", dt["text"])
#
#     out_spo = entity_describe_analysis(sentence_list)
#
#     for sentence, spo in out_spo:
#         print(sentence)
#         print(spo)
