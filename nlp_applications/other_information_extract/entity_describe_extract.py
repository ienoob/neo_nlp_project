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


data_list = [
    {"entity": "数学", "describe_list": ["利用符号语言研究数量、结构、变化以及空间等概念的一门学科", "形式科学的一种"], "text": "数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科，从某种角度看属于形式科学的一种。数学透过抽象化和逻辑推理的使用，由计数、计算、量度和对物体形状及运动的观察而产生。"},
    {"entity": "哲学", "describe_list": ["研究普遍的、根本的问题的学科", "个人或团体的最基本信仰、概念或态度"], "text": "哲学（）是研究普遍的、根本的问题的学科，包括存在、知识、价值、理智、心灵、语言等领域。哲学与其他学科的不同是其批判的方式、通常是系统化的方法，并以理性论证为基础。在日常用语中，其也可被引申为个人或团体的最基本信仰、概念或态度。"},
    {"entity": "文学", "describe_list": ["任何单一的书面作品", "一种艺术形式", "具有艺术或智力价值的任何单一作品"], "text": "文学（），在最广泛的意义上，是任何单一的书面作品。更严格地说，文学写作被认为是一种艺术形式，或被认为具有艺术或智力价值的任何单一作品，通常是由于以不同于普通用途的方式部署语言。"},
    {"entity": "历史", "describe_list": ["人类社会过去的事件和行动", "对这些事件行为有系统的记录、诠释和研究", "人类精神文明的重要成果", "对过去事件的记录和研究"], "text": "历史（现代汉语词汇，古典文言文称之为史），指人类社会过去的事件和行动，以及对这些事件行为有系统的记录、诠释和研究。历史可提供今人理解过去，作为未来行事的参考依据，与伦理、哲学和艺术同属人类精神文明的重要成果。"},
    {"entity": "计算机科学", "describe_list": ["系统性研究信息与计算的理论基础", "计算机系统中如何与应用的实用技术的学科", "对那些创造、描述以及转换信息的算法处理的系统研究"], "text": "计算机科学（，有时缩写为）是系统性研究信息与计算的理论基础以及它们在计算机系统中如何与应用的实用技术的学科。 它通常被形容为对那些创造、描述以及转换信息的算法处理的系统研究。计算机科学包含很多分支领域；有些强调特定结果的计算，比如计算机图形学；而有些是探讨计算问题的性质，比如计算复杂性理论；还有一些领域专注于怎样实现计算，比如程式语言理论是研究描述计算的方法，而程式设计是应用特定的程式语言解决特定的计算问题，人机交互则是专注于怎样使计算机和计算变得有用、好用，以及随时随地为人所用。"},
    {"entity": "民族", "describe_list": ["人", "具有十分丰富而复杂的内涵", "族群", "国族"], "text": "在汉语中，民族一词具有十分丰富而复杂的内涵，可以表达多种近似而不同的概念。词汇本身歧义较多，概念和用法受到政治的较大影响，这些义项之间容易相互混淆。在不同的学科中，对于民族的范畴与用法也有许多歧异。在学术上，族群比民族的概念更宽泛。而在汉语实际使用中，民族可以被表示为包括族群、国族在内的多种含义。民族一词在中英翻译时也十分容易混淆。Ethnic group和Nation经常被翻译为民族，然而更精确地应分别译为译为族群和国族。"},
    {"entity": "戏剧", "describe_list": ["演员将某个故事或情境，以对话、歌唱或动作等方式所表演出来的艺术"]},
    {"entity": "电影", "describe_list": ["一种表演艺术、视觉艺术及听觉艺术，利用胶卷、录影带或数位媒体将影像和声音捕捉起来，再加上后期的编辑工作而成。电影中看起来连续的画面", "由一帧帧单独的照片构成的"]},
    {"entity": "音乐", "describe_list": ["指任何以声音组成的艺术"]},
    {"entity": "经济学", "describe_list": ["一门对产品和服务的生产、分配以及消费进行研究的社会科学"]},
    {"entity": "政治学", "describe_list": ["一门以研究政治行为、政治体制以及政治相关领域为主的社会科学学科"]},
    {"entity": "法学", "describe_list": ["社会科学中一门特殊的学科"]},
    {"entity": "社会学", "describe_list": ["一门研究社会的学科"]},
    {"entity": "军事学", "describe_list": []},
    {"entity": "信息学", "describe_list": ["以信息为研究对象，利用计算机及其程序设计等技术为研究工具来分析问题、解决问题的学问，是以扩展人类的信息功能为主要目标的一门综合性学科"]},
    {"entity": "物理学", "describe_list": ["研究物质、能量的本质与性质的自然科学", "自然科学中最基础的学科之一", "一种实验科学"]}
]

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
