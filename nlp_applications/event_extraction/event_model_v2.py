#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument


sample_path = "D:\\data\\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64
event_num = len(bd_data_loader.event2id)
batch_num = 10
argument_num = len(bd_data_loader.argument_role2id)


class DataIter(object):

    def __init__(self):
        pass

class EventModelV2(tf.keras.Model):

    def __init__(self):
        super(EventModelV2, self).__init__()
