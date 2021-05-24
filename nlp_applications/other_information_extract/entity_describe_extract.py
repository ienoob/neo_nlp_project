#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/17 23:12
    @Author  : jack.li
    @Site    : 
    @File    : entity_describe_extract.py

    任务名称：实体 实体描述分析，目的是抽取出文本中的实体以及对应的实体描述。

"""
import time
import re
from ltp import LTP
from nlp_applications.data_loader import load_json_line_data
import multiprocessing
from joblib import Parallel, delayed

data_path = "D:\data\语料库\wiki_zh_2019\wiki_zh\AA\\wiki_00"

data = load_json_line_data(data_path)


class EntityDescribeExtractByRoleAnalysis(object):
    """
        基于ltp 角色分析进行实体和实体描述信息抽取
    """

    def __init__(self):
        self.ltp = LTP("tiny")

    def single_sentence(self, input_sentence, ind=0):
        # ltp = LTP()
        seg, hidden = self.ltp.seg([input_sentence])
        words = seg[ind]

        pos = self.ltp.pos(hidden)[ind]
        roles = self.ltp.srl(hidden, keep_empty=False)[ind]

        # print(words)
        # print(roles)
        filter_p = {"是", "为"}
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

            sub_value = words[sub[1]:sub[2] + 1]

            obj_value = words[obj[1]:obj[2] + 1]
            # print("".join(sub_value), p_value, "".join(obj_value))
            spo_list.append(("".join(sub_value), p_value, "".join(obj_value)))

        return spo_list

    def extract_info(self, input_sentence_list):
        spo_res = []
        # input_sentence_list = [sentence for sentence in input_sentence_list if sentence.strip()]
        # seg, hidden = self.ltp.seg(input_sentence_list)
        for i, sentence in enumerate(input_sentence_list):
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            # if len(seg[i]) < 3:
            #     continue
            out_spo_list = self.single_sentence(sentence)

            if out_spo_list:
                spo_res.append((sentence, out_spo_list))
        return spo_res

    def multi_extract_info(self, input_sentence_list):
        pool = multiprocessing.Pool(processes=3)
        spo_res = []
        for i, sentence in enumerate(input_sentence_list):
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            out_spo_list = pool.apply_async(self.single_sentence, (sentence,))
            # out_spo_list = self.single_sentence(sentence)
            spo_res.append(out_spo_list)
            # spo_res.append((sentence, out_spo_list))
        pool.close()
        pool.join()

        spo_res = [spo.get() for i, spo in enumerate(spo_res)]
        return spo_res



# input_data = "文学（），在最广泛的意义上，是任何单一的书面作品。"
# pattern = "^(.+?)是(.+?)[,，。.]"
#
# # print(re.findall(pattern, input_data))
#
# filter_p = {"是", "为"}
#
#
# contents = "文学批评是指文学批评者对其他人作品的评论和评估，有时也会用来改进及提升文学作品。"
# sentence_list = contents.split("。")
# single_sentence(sentence)


def multi_process(processes_num=4):
    ede_model = EntityDescribeExtractByRoleAnalysis()
    pool = multiprocessing.Pool(processes=processes_num)
    result = []
    for i, dt in enumerate(data):
        if i >= 5:
            break
        print(dt["title"])
        sentence_list = re.split("[。\n]", dt["text"])

        print(len(sentence_list))

        out_spo = pool.apply_async(ede_model.extract_info, (sentence_list,))
        result.append(out_spo)

    for res in result:
        print(":::", res.get())




if __name__ == "__main__":
    ede_model = EntityDescribeExtractByRoleAnalysis()
    # out_spo = ede_model.extract_info(sentence_list)
    #
    # for sentence, spo in out_spo:
    #     print(sentence)
    #     print(spo)
    start_a_time = time.time()

    # out_spo_list = Parallel(n_jobs=2)(delayed(self.single_sentence)(self, sentence) for name, group in tqdm(df))
    # pool = multiprocessing.Pool(processes = 3)
    # manager = multiprocessing.Manager()
    result = []
    count = 0
    for i, dt in enumerate(data):
        if i >= 5:
            break
        print(dt["title"])
        # print(dt["text"])
        sentence_list = re.split("[。\n]", dt["text"])

        print(len(sentence_list))
        count += len(sentence_list)

        start_time = time.time()
        # out_spo = pool.apply_async(ede_model.extract_info, (sentence_list,))
        #
        # out_spo = ede_model.multi_extract_info(sentence_list)
        out_spo = ede_model.extract_info(sentence_list)
        result.append(out_spo)
        #
        # for sentence, spo in out_spo:
        #     print(sentence)
        #     print(spo)

        cost_time = time.time()-start_time
        print("cost {} s".format(cost_time))
        print("{} /s".format(len(sentence_list)/cost_time))

    # pool.close()
    # pool.join()

    print("all cost {}s".format(time.time()-start_a_time))
    print("all {}/s".format(count/(time.time() - start_a_time)))

    for res in result:
        print(res)
    #     print(":::", res.get())
