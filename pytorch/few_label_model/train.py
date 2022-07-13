#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
import torch
import torch.autograd as autograd
from models import Generator, Discriminator
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, BaseDataIterator


sample_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取"
bd_data_loader = LoaderBaiduDueeV1(sample_path)
selector = Generator()
discriminator = Discriminator()


def sequence_padding(inputs, length=None, padding=0, is_float=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return torch.tensor(out_tensor)

class DataIter(BaseDataIterator):
    def __init__(self, input_data_ld):
        super(DataIter, self).__init__(input_data_ld)

    def single_preprocess(self, document: EventDocument):
        sentence_id = document.text_id
        sentence_len = len(sentence_id)
        sentence_pos = [i+1 for i in range(sentence_len)]

        event_rd = np.random.choice(document.event_list)
        # event_trigger = event_rd.trigger
        event_trigger_loc = event_rd.trigger_start_index
        tag_local = [sentence_id[event_trigger_loc-1], sentence_id[event_trigger_loc], sentence_id[event_trigger_loc+1]]
        tag_label = event_rd.event_id
        tag_mask_l = [1 if i < event_trigger_loc else 0 for i in range(sentence_len)]
        tag_mask_r = [1 if i >= event_trigger_loc else 0 for i in range(sentence_len)]

        return {
            "sentence_id": sentence_id,
            "sentence_pos": sentence_pos,
            "tag_local": tag_local,
            "tag_label": tag_label,
            "tag_mask_l": tag_mask_l,
            "tag_mask_r": tag_mask_r
        }

    def padding_batch_data(self, input_batch_data):
        batch_sentence_id = []
        batch_sentence_pos = []
        batch_tag_local = []
        batch_tag_label = []
        batch_tag_mask_l = []
        batch_tag_mask_r = []

        max_len = 0
        for data in input_batch_data:
            batch_sentence_id.append(data["sentence_id"])
            batch_sentence_pos.append(data["label_id"])
            batch_tag_local.append(data["tag_local"])
            batch_tag_label.append(data["tag_mask_l"])
            batch_tag_mask_l.append(data["tag_mask_l"])
            batch_tag_mask_r.append(data["tag_mask_r"])
            max_len = max(max_len, len(data["sentence_id"]))

        batch_sentence_id = autograd.Variable(sequence_padding(batch_sentence_id, length=max_len)).long()
        batch_sentence_pos = autograd.Variable(sequence_padding(batch_sentence_pos, length=max_len)).long()
        batch_tag_local = sequence_padding(batch_tag_local, length=max_len).long()
        batch_tag_label = autograd.Variable(batch_tag_label).long()
        batch_tag_mask_l = sequence_padding(batch_tag_mask_l, length=max_len).long()
        batch_tag_mask_r = autograd.Variable(batch_tag_mask_r).long()

        return {
            "sentence_id": batch_sentence_id,
            "sentence_pos": batch_sentence_pos,
            "tag_local": batch_tag_local,
            "tag_label": batch_tag_label,
            "tag_mask_l": batch_tag_mask_l,
            "tag_mask_r": batch_tag_mask_r
        }

if __name__ == "__main__":
    batch_num = 2
    data_iter = DataIter(bd_data_loader)

    for batch_data in data_iter.train_iter(batch_num):
        uwords, upos, uloc, umaskL, umaskR, ulabel, utimes = \
            batch_data["sentence_id"], batch_data["sentence_pos"], batch_data["tag_local"], batch_data["tag_mask_l"], batch_data["tag_mask_r"]
        break











