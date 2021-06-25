#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument, BaseDataIterator
from nlp_applications.event_extraction.event_model_v2 import EventModelV2


sample_path = "D:\\data\\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64
event_num = len(bd_data_loader.event2id)
batch_num = 10
argument_num = len(bd_data_loader.argument_role2id)
event2argument_dict = bd_data_loader.event2argument


class DataIter(BaseDataIterator):

    def __init__(self, data_loader):
        super(DataIter, self).__init__(data_loader)
        self.event_type_num = event_num
        self.event2argument_bio = dict()
        for k, argument in event2argument_dict.items():
            argument_d = {"O": 0}
            for arg in argument:
                argument_d["B-"+arg] = len(argument_d)
                argument_d["I-" + arg] = len(argument_d)
            self.event2argument_bio[k] = argument_d

    def single_doc_processor(self, doc: EventDocument):
        text_id = doc.text_id

        event_label = np.zeros(self.event_type_num)
        for event in doc.event_list:
            event_label[event.id] = 1

        rd_event = np.random.choice(doc.event_list)
        rd_event_type = rd_event.id
        argument_role2id = self.event2argument_bio[rd_event_type]
        rd_argument = np.zeros(len(text_id))
        for arg in rd_event.arguments:
            rd_argument[arg.start] = argument_role2id["B-"+arg.role]
            for iv in range(arg.start+1, arg.start+len(arg.argument)-1):
                rd_argument[iv] = argument_role2id["I-"+arg.role]

        return {
            "char_id": text_id,
            "event_label": event_label,
            "rd_event_type": [rd_event_type],
            "rd_argument": rd_argument
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_id = []
        batch_event_label = []
        batch_rd_event_type = []
        batch_rd_argument = []

        for data in input_batch_data:
            batch_char_id.append(data["char_id"])
            batch_event_label.append(data["event_label"])
            batch_rd_event_type.append(data["rd_event_type"])
            batch_rd_argument.append(data["rd_argument"])
        batch_char_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_id, padding="post")
        batch_event_label = tf.cast(batch_event_label, tf.int32)
        batch_rd_event_type = tf.cast(batch_rd_event_type, tf.int32)
        batch_rd_argument = tf.keras.preprocessing.sequence.pad_sequences(batch_rd_argument, padding="post")

        return {
            "char_id": batch_char_id,
            "event_label": batch_event_label,
            "rd_event_type": batch_rd_event_type,
            "rd_argument": batch_rd_argument
        }


char_size = len(bd_data_loader.char2id)
char_embed = 64
lstm_size = 64
event_class = len(bd_data_loader.event2id)
event_embed_size = 64

data_iter = DataIter(bd_data_loader)
event2argument = data_iter.event2argument_bio

model = EventModelV2(char_size, char_embed, lstm_size, event_class, event_embed_size, event2argument)

optimizer = tf.keras.optimizers.Adam()
loss_func1 = tf.keras.losses.SparseCategoricalCrossentropy()

def loss_func(input_argument, input_argument_logits, mask):
    loss_func2 = tf.keras.losses.SparseCategoricalCrossentropy()

    return loss_func2(input_argument, input_argument_logits, sample_weight=mask)


@tf.function()
def train_step(input_char, input_event_label, input_rd_event, input_rd_argument):
    with tf.GradientTape() as tape:
        event_logits, rd_argument_logits, mask = model(input_char, input_rd_event)

        loss1 = loss_func1(input_event_label, event_logits)
        loss2 = loss_func(input_rd_argument, rd_argument_logits, mask)

        lossv = loss1 + loss2

    variables = model.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv


if __name__ == "__main__":
    batch_num = 10

    epoch = 10
    for i, batch_data in enumerate(data_iter.train_iter(batch_num)):
        loss_value = train_step(batch_data["char_id"], batch_data["event_label"],
                                batch_data["rd_event_type"], batch_data["rd_argument"])

        print(loss_value)
        break
