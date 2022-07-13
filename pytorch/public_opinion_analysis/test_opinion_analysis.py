#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import hanlp
from ltp import LTP
data_content = "格隆汇2月11日丨万马科技(300698.SZ)低开低走，盘中低见21.39元创去年9月15日以来近5个月新低，现跌4.59%报21.4元，暂成交3455万元，最新市值28亿元。万马科技昨日晚间公布，公司近日收到公司特定股东杨义谦遵照其在《首次公开发行股票并在创业板上市招股说明书》等资料中的相关承诺提交的《关于股份减持计划的告知函》，减持其所持公司无限售条件流通股不超过268万股，占公司总股本的2.00%。"

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
res = HanLP(data_content.split("。"))

print(res["tok/fine"])
onto_ner = res["ner/ontonotes"]
print(len(onto_ner))

for ner in onto_ner:
    print(ner)

# 获取person 和 company
# 获取金融事件类型

# ltp = LTP()
# seg, hidden = ltp.seg(["格隆汇2月11日丨万马科技(300698.SZ)低开低走，盘中低见21.39元创去年9月15日以来近5个月新低，现跌4.59%报21.4元，暂成交3455万元，最新市值28亿元。"])
# # pos = ltp.pos(hidden)
# ner = ltp.ner(hidden)
#
# print(ner)

