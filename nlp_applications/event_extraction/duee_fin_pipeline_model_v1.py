#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/7/31 23:56
    @Author  : jack.li
    @Site    : 
    @File    : duee_fin_pipeline_model_v1.py

"""
import numpy as np
from nlp_applications.data_loader import LoaderBaiduDueeFin, EventDocument, Event, Argument, BaseDataIterator
from nlp_applications.ner.evaluation import extract_entity

sample_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\\篇章级事件抽取"
bd_data_loader = LoaderBaiduDueeFin(sample_path)
max_len = 256
print(bd_data_loader.event2id)
print(bd_data_loader.argument_role2id)


class DataIter(BaseDataIterator):

    def __init__(self, input_data_loader):
        super(DataIter, self).__init__(input_data_loader)
        self.event_argument2id = self.data_loader.event_argument2id
        self.id2event = self.data_loader.id2event
        self.bio = {"O": 0}
        for k in self.event_argument2id:
            self.bio["{}-{}".format("B", k)] = len(self.bio)
            self.bio["{}-{}".format("I", k)] = len(self.bio)

    def single_doc_processor(self, doc: EventDocument):

        text_id = doc.title_id+doc.text_id
        text_all = doc.title+doc.text
        label_value = np.zeros(len(text_id))
        label_value_ = ["O" for _ in text_id]

        for event in doc.event_list:
            event_type = self.id2event[event.id]
            for argument in event.arguments:
                event_arg_kwy = "{}_{}".format(event_type, argument.role)
                arg = argument.argument
                if argument.role == "环节":
                    continue
                try:
                    arg_loc = text_all.index(arg)
                except Exception as e:
                    print(text_all, arg, argument.role)
                    raise e
                for i, _ in enumerate(arg):
                    if i == 0:
                        bio_event_arg_kwy = "B-{}".format(event_arg_kwy)
                        label_value[arg_loc+i] = self.bio[bio_event_arg_kwy]
                        label_value_[arg_loc+i] = "B-{}".format(label_value[arg_loc+i])
                    else:
                        bio_event_arg_kwy = "I-{}".format(event_arg_kwy)
                        label_value[arg_loc + i] = self.bio[bio_event_arg_kwy]
                        label_value_[arg_loc + i] = "I-{}".format(label_value[arg_loc + i])

        return {
            "text_id": text_id,
            "text": text_all,
            "label_id": label_value,
            "label_value": label_value_
        }

    def padding_batch_data(self, input_batch_data):
        batch_text_id = []
        batch_label_id = []

        for data in input_batch_data:
            batch_text_id.append(data["text_id"])
            batch_label_id.append(data["label_id"])

        return {
            "text_id": batch_text_id,
            "label_id": batch_label_id
        }

    def get_crf_data(self, mode="train"):
        if mode == "train":
            doc_list = self.data_loader.documents
        else:
            doc_list = self.data_loader.documents
        text_list = []
        label_value_list = []

        for doc in doc_list:
            bd_data = self.single_doc_processor(doc)
            text_list.append(bd_data["text"])
            label_value_list.append(bd_data["label_value"])

        return text_list, label_value_list


data_iter = DataIter(bd_data_loader)

text_list, label_value_list = data_iter.get_crf_data()
dev_text_list, dev_label_list = data_iter.get_crf_data("dev")

# 测试crf 模型
from nlp_applications.ner.crf_model import CRFNerModel, sent2features,  sent2labels
from nlp_applications.ner.evaluation import metrix_v2

print(type(text_list[0]))
crf_mode = CRFNerModel()
X_train = [sent2features(s) for s in text_list]
y_train = [sent2labels(s) for s in label_value_list]

X_dev = [sent2features(s) for s in dev_text_list]
y_dev = [sent2labels(s) for s in dev_label_list]

print(len(X_train))
crf_mode.fit(X_train, y_train)

predict_labels = crf_mode.predict_list(X_dev)

print(metrix_v2(y_dev, predict_labels))

# X_test = [sent2features(s) for s in msra_data.test_sentence_list]
# y_test = [sent2labels(s) for s in msra_data.test_tag_list]

