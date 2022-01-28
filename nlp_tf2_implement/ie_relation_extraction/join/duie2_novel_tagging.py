#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/31 22:19
    @Author  : jack.li
    @Site    : 
    @File    : duie2_novel_tagging.py

"""
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from nlp_applications.ner.evaluation import extract_entity
from nlp_applications.ie_relation_extraction.join.novel_tagging import NovelTaggingModelPointerNet


data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)
triple_regularity = data_loader.triple_set
relation_bio_encoder = {"O": 0}

for i in range(1, len(data_loader.relation2id)):
    relation_bio_encoder["B-1-{}".format(i)] = len(relation_bio_encoder)
    relation_bio_encoder["B-2-{}".format(i)] = len(relation_bio_encoder)
    relation_bio_encoder["I-1-{}".format(i)] = len(relation_bio_encoder)
    relation_bio_encoder["I-2-{}".format(i)] = len(relation_bio_encoder)
relation_bio_id2encoder = {v:k for k, v in relation_bio_encoder.items()}



