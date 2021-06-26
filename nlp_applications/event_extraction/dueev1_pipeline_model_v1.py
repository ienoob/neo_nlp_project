#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/26 16:59
    @Author  : jack.li
    @Site    : 
    @File    : dueev1_pipeline_model_v1.py

"""

import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument, BaseDataIterator
from nlp_applications.event_extraction.pipeline_model_v1 import EventTypeClassifier, EventArgument

sample_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)
event2argument_dict = bd_data_loader.event2argument
event_num = len(bd_data_loader.event2id)


class DataIter(BaseDataIterator):

    def __init__(self, data_loader):
        super(DataIter, self).__init__(data_loader)
        self.trigger_bio2id = {
            "O": 0
        }
        for en in range(event_num):
            self.trigger_bio2id["B-{}".format(en)] = len(self.trigger_bio2id)
            self.trigger_bio2id["I-{}".format(en)] = len(self.trigger_bio2id)

    def single_doc_processor(self, doc: EventDocument):
        text_id = doc.text_id

        event_label = np.zeros(event_num)
        event_trigger = np.zeros(len(text_id))
        for event in doc.event_list:
            event_label[event.id] = 1
            trigger_start = event.trigger_start
            trigger_end = event.trigger_start + len(event.trigger)
            event_trigger[trigger_start] = self.trigger_bio2id["B-{}".format(event.id)]
            event_trigger[trigger_start+1:trigger_end] = self.trigger_bio2id["I-{}".format(event.id)]

        return {
            "char_id": text_id,
            "event_label": event_label,
            "event_trigger": event_trigger
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_id = []
        batch_event_label = []
        batch_event_trigger = []

        for data in input_batch_data:
            batch_char_id.append(data["char_id"])
            batch_event_label.append(data["event_label"])
            batch_event_trigger.append(data["event_trigger"])

        batch_char_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_id, padding="post")
        batch_event_label = tf.cast(batch_event_label, tf.int32)
        batch_event_trigger = tf.keras.preprocessing.sequence.pad_sequences(batch_event_trigger, padding="post")
        return {
            "char_id": batch_char_id,
            "event_label": batch_event_label,
            "event_trigger": batch_event_trigger
        }


class DataIterArgument(BaseDataIterator):

    def __init__(self, data_loader):
        super(DataIterArgument, self).__init__(data_loader)
        self.event_type_num = event_num
        self.event2argument_bio = dict()
        for k, argument in event2argument_dict.items():
            argument_d = {"O": 0}
            for arg in argument:
                argument_d["B-"+arg] = len(argument_d)
                argument_d["I-" + arg] = len(argument_d)
            self.event2argument_bio[k] = argument_d

    def single_doc_processor(self, doc: EventDocument, event_type=0):
        text_id = doc.text_id

        event_list = []
        for event in doc.event_list:
            if event.id != event_type:
                continue
            event_list.append(event)

        if len(event_list) == 0:
            return dict()

        rd_event = np.random.choice(event_list)
        rd_event_type = rd_event.id
        argument_role2id = self.event2argument_bio[rd_event_type]
        rd_argument = np.zeros(len(text_id))
        for arg in rd_event.arguments:
            rd_argument[arg.start] = argument_role2id["B-"+arg.role]
            for iv in range(arg.start+1, arg.start+len(arg.argument)-1):
                rd_argument[iv] = argument_role2id["I-"+arg.role]

        return {
            "rd_event_type": [rd_event_type],
            "rd_argument": rd_argument
        }

    def padding_batch_data(self, input_batch_data):
        batch_rd_event_type = []
        batch_rd_argument = []

        for data in input_batch_data:
            batch_rd_event_type.append(data["rd_event_type"])
            batch_rd_argument.append(data["rd_argument"])
        batch_rd_event_type = tf.cast(batch_rd_event_type, tf.int32)
        batch_rd_argument = tf.keras.preprocessing.sequence.pad_sequences(batch_rd_argument, padding="post")

        return {
            "rd_event_type": batch_rd_event_type,
            "rd_argument": batch_rd_argument
        }


char_size = len(bd_data_loader.char2id)
char_embed = 64
lstm_size = 64
event_class = len(bd_data_loader.event2id)
event_embed_size = 64

data_iter = DataIter(bd_data_loader)

trigger_num = len(data_iter.trigger_bio2id)
event_model = EventTypeClassifier(char_size, char_embed, lstm_size, event_class, trigger_num)

optimizer = tf.keras.optimizers.Adam()
loss_func1 = tf.keras.losses.BinaryCrossentropy()

def loss_func2(true_trigger, logits_trigger, mask=None):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy()

    return cross_func(true_trigger, logits_trigger, sample_weight=mask)


@tf.function(experimental_relax_shapes=True)
def train_step(input_x, input_event, input_trigger):
    with tf.GradientTape() as tape:
        trigger_logits, event_logits, mask = event_model(input_x)

        lossv = loss_func1(input_event, event_logits) + loss_func2(input_trigger, trigger_logits)

    variable = event_model.trainable_variables
    gradients = tape.gradient(lossv, variable)
    optimizer.apply_gradients(zip(gradients, variable))

    return lossv


epoch = 10
batch_num = 32
for e in range(epoch):

    for bi, batch_data in enumerate(data_iter.train_iter(batch_num)):
        loss_value = train_step(batch_data["char_id"], batch_data["event_label"], batch_data["event_trigger"])

        if bi % 100 == 0:
            print("epoch {0} batch {1} loss value {2}".format(e, bi, loss_value))

