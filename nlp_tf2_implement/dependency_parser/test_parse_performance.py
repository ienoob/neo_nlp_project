#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    这里测试句法\语义分析库的性能
"""
import os
import time
import hanlp
from pyhanlp import HanLP
from utils.neo_function import split_str

weibo_path = "D:\data\语料库\weibo_2019-05-18_10.30.41.txt\weibo_2019-05-18_10.30.41"
wiki_path = "D:\data\语料库\wiki_zh_2019\wiki_zh\AA"


def generator_weibo_list(i_path):
    file_list = os.listdir(i_path)
    for file in file_list:
        weibo_file_path = i_path + "\\" + file
        with open(weibo_file_path, "r", encoding="utf-8") as f:
            data = f.read()

            for weibo_one in data.split("\n"):
                weibo_one = weibo_one.strip()
                if not weibo_one:
                    continue
                weibo_cut = split_str(weibo_one, {"。", "？", "！", "!", "?", "\r", "\n", " ", "."})
                for sentence in weibo_cut:
                    yield sentence


generator = generator_weibo_list(weibo_path)
weib_sentence = []
i = 0
max_num = 2000
for sentence in generator:
    weib_sentence.append(sentence)
    i += 1
    if i > max_num:
        break
weib_sentence_size = sum([len(weibo)*3 for weibo in weib_sentence])
print("data sentence num {}".format(len(weib_sentence)))
print("data size {}".format(weib_sentence_size))


def test_ltp_single():
    from ltp import LTP
    ltp = LTP("tiny")
    res = []
    for sentence in weib_sentence:
        seg, hidden = ltp.seg([sentence])
        pos = ltp.pos(hidden)[0]
        res.append(pos)
        roles = ltp.srl(hidden, keep_empty=False)
        res.append(roles)


def test_ltp_batch():
    from ltp import LTP
    ltp = LTP("tiny")
    # for sentence in weib_sentence:
    seg, hidden = ltp.seg(weib_sentence)
    pos = ltp.pos(hidden)[0]
    roles = ltp.srl(hidden, keep_empty=False)


hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)


def test_hanlp_single():
    for sentence in weib_sentence:
        document = hanlp_model([sentence])


def test_hanlp_batch():
    document = hanlp_model(weib_sentence)


def test_hanlpV1_single():
    for sentence in weib_sentence:
        document = HanLP.parseDependency(sentence)


# print("test start")

start = time.time()

test_ltp_single()

end = time.time()
cost_time = end-start
print("cost {} s".format(cost_time))
print("sentence/s {}".format(len(weib_sentence)/cost_time))
print("byte/s {}".format(weib_sentence_size/cost_time))

"""
    ltp single 
    data sentence num 2001
    data size 108996
    cost 26.273699283599854 s
    sentence/s 76.15981207674977
    byte/s 4148.483196960229

    ltp batch
    data sentence num 2001
    data size 108996
    cost 42.22348213195801 s
    sentence/s 47.39069112647836
    byte/s 2581.4071814201075
    
    hanlp single
    data sentence num 2001
    data size 108996
    cost 155.52888703346252 s
    sentence/s 12.86577714382717
    byte/s 700.8087184250805
    
    data sentence num 2001
    data size 108996
    cost 67.15310049057007 s
    sentence/s 29.797581725671613
    byte/s 1623.0970603554738
    
    
    hanlp v1 single
    data sentence num 2001
    data size 108996
    cost 54.05655074119568 s
    sentence/s 37.0167902421319
    byte/s 2016.3328681816133
"""
