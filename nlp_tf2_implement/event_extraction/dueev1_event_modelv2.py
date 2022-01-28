#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import logging
import numpy as np
import tensorflow as tf
from tf2.python.custom_schedule import CustomSchedule
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument, BaseDataIterator
from nlp_applications.event_extraction.event_model_v2 import EventModelV2
from evaluation import extract_entity, eval_metrix


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sample_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取"
bd_data_loader = LoaderBaiduDueeV1(sample_path)

vocab_size = len(bd_data_loader.char2id)
event_num = len(bd_data_loader.event2id)

argument_num = len(bd_data_loader.argument_role2id)
event2argument_dict = bd_data_loader.event2argument
event2argument_list = bd_data_loader.event2argument_list

logger.info("event num {}".format(event_num))
logger.info("argument num {}".format(argument_num))


class DataIter(BaseDataIterator):

    def __init__(self, data_loader):
        super(DataIter, self).__init__(data_loader)
        self.event_type_num = event_num
        self.event2argument_bio = dict()
        self.argument_bio = {"O": 0}

        for a in range(argument_num):
            self.argument_bio["B-{}".format(a)] = len(self.argument_bio)
            self.argument_bio["I-{}".format(a)] = len(self.argument_bio)

        self.argument_id2bio = {v:k for k, v in self.argument_bio.items()}

        self.event2argument_mask = dict()
        for event_type in range(1, event_num):
            rd_argument_mask = np.ones((1, len(self.argument_bio))) * (-1e9)
            rd_argument_mask[0][0] = 0
            for arg_id in event2argument_list[event_type]:
                b_id = self.argument_bio["B-{}".format(arg_id)]
                rd_argument_mask[0][b_id] = 0
                i_id = self.argument_bio["I-{}".format(arg_id)]
                rd_argument_mask[0][i_id] = 0
            self.event2argument_mask[event_type] = rd_argument_mask
        # for k, argument in event2argument_dict.items():
        #     argument_d = {"O": 0}
        #     for arg in argument:
        #         argument_d["B-"+arg] = len(argument_d)
        #         argument_d["I-" + arg] = len(argument_d)
        #     self.event2argument_bio[k] = argument_d

    def single_doc_processor(self, doc: EventDocument):
        text_id = doc.text_id

        event_label = np.zeros(self.event_type_num)
        event_true_res = []
        for event in doc.event_list:
            event_label[event.id] = 1
            event_arg = [(arg.start, arg.start+len(arg.argument), arg.role_id) for arg in event.arguments]
            event_true_res.append((event.id, event_arg))

        rd_event = np.random.choice(doc.event_list)
        rd_event_type = rd_event.id
        rd_argument = np.zeros(len(text_id))
        rd_argument_mask = np.ones((1, len(self.argument_bio)))*(-1e9)
        rd_argument_mask[0][0] = 0
        for arg_id in event2argument_list[rd_event_type]:
            b_id = self.argument_bio["B-{}".format(arg_id)]
            rd_argument_mask[0][b_id] = 0
            i_id = self.argument_bio["I-{}".format(arg_id)]
            rd_argument_mask[0][i_id] = 0
        for arg in rd_event.arguments:
            rd_argument[arg.start] = self.argument_bio["B-{}".format(arg.role_id)]
            for iv in range(arg.start+1, arg.start+len(arg.argument)):
                rd_argument[iv] = self.argument_bio["I-{}".format(arg.role_id)]

        return {
            "char_id": text_id,
            "event_label": event_label,
            "rd_event_type": [rd_event_type],
            "rd_argument": rd_argument,
            "rd_argument_mask": rd_argument_mask,
            "event_true_res": event_true_res
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_id = []
        batch_event_label = []
        batch_rd_event_type = []
        batch_rd_argument = []
        batch_rd_argument_mask = []
        batch_event_true_res = []

        for data in input_batch_data:
            batch_char_id.append(data["char_id"])
            batch_event_label.append(data["event_label"])
            batch_rd_event_type.append(data["rd_event_type"])
            batch_rd_argument.append(data["rd_argument"])
            batch_rd_argument_mask.append(data["rd_argument_mask"])
            batch_event_true_res.append(data["event_true_res"])
        batch_char_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_id, padding="post")
        batch_event_label = tf.cast(batch_event_label, tf.int32)
        batch_rd_event_type = tf.cast(batch_rd_event_type, tf.int32)
        batch_rd_argument = tf.keras.preprocessing.sequence.pad_sequences(batch_rd_argument, padding="post")
        batch_rd_argument_mask = tf.cast(batch_rd_argument_mask, tf.float32)

        return {
            "char_id": batch_char_id,
            "event_label": batch_event_label,
            "rd_event_type": batch_rd_event_type,
            "rd_argument": batch_rd_argument,
            "rd_argument_mask": batch_rd_argument_mask,
            "event_true_res": batch_event_true_res
        }

    def dev_iter(self, input_batch_num):
        c_batch_data = []
        for doc in self.data_loader.dev_documents:
            c_batch_data.append(self.single_doc_processor(doc))
            if len(c_batch_data) == input_batch_num:
                yield self.padding_batch_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_batch_data(c_batch_data)


char_size = len(bd_data_loader.char2id)
batch_num = 10
char_embed = 64
embed_size = 64
lstm_size = 64
event_class = len(bd_data_loader.event2id)
event_embed_size = 64

data_iter = DataIter(bd_data_loader)
event2argument_mask = data_iter.event2argument_mask
argument_id2bio = data_iter.argument_id2bio

model = EventModelV2(char_size, char_embed, lstm_size, event_class, event_embed_size, len(data_iter.argument_bio), event2argument_mask)

lr = CustomSchedule(embed_size)
optimizer = tf.keras.optimizers.Adam(lr)


def loss_func1(true_value, pre_value):
    binary_func = tf.keras.losses.BinaryCrossentropy()
    mask = tf.greater(true_value, 0)
    mask = tf.where(mask, 7.0, 1.0)
    true_value = tf.expand_dims(true_value, axis=-1)
    pre_value = tf.expand_dims(pre_value, axis=-1)
    return binary_func(true_value, pre_value, sample_weight=mask)


def loss_func(input_argument, input_argument_logits, mask):
    cross_func2 = tf.keras.losses.SparseCategoricalCrossentropy()

    return cross_func2(input_argument, input_argument_logits, sample_weight=mask)


@tf.function(experimental_relax_shapes=True)
def train_step(input_char, input_event_label, input_rd_event, input_rd_argument, input_rd_argument_mask):
    with tf.GradientTape() as tape:
        event_logits, rd_argument_logits, mask = model(input_char, input_rd_event, input_rd_argument_mask, training=True)

        loss1 = loss_func1(input_event_label, event_logits)
        loss2 = loss_func(input_rd_argument, rd_argument_logits, mask)

        lossv = loss1 + loss2

    variables = model.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv


def iter_all(input_list, i, cache):
    if i == len(input_list):
        yield cache
    else:
        for role in input_list[i]:
            yield from iter_all(input_list, i+1, cache+[role])


def evaluation(input_batch_data, model):
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0
    batch_event_res = model(input_batch_data["char_id"])
    batch_event_true_res = input_batch_data["event_true_res"]
    for bi, event_row in enumerate(batch_event_res):
        event_row_res = []
        event_true_d = dict()
        true_num += len(batch_event_true_res[bi])
        for e_i, e_arg in batch_event_true_res[bi]:
            event_true_d.setdefault(e_i, [])
            e_arg.sort()
            event_true_d[e_i].append(e_arg)
        for event_id, event_arg in event_row:
            event_arg = [argument_id2bio[ea] for ea in event_arg]
            extract_arg = extract_entity(event_arg)
            extract_arg = [(x, y, int(z)) for x, y, z in extract_arg]
            extract_arg_dict_by_role = dict()
            for x, y, role_t in extract_arg:
                extract_arg_dict_by_role.setdefault(role_t, [])
                extract_arg_dict_by_role[role_t].append((x, y, role_t))

            extract_arg_list = [v for _, v in extract_arg_dict_by_role.items()]
            if event_id in event_true_d:
                for extract_arg in iter_all(extract_arg_list, 0, []):
                    if extract_arg in event_true_d[event_id]:
                        hit_num += 1

            event_row_res.append((event_id, extract_arg))

            pre_num += 1

    print(hit_num, pre_num, true_num)
    return {"hit_num": hit_num,
            "pred_num": pre_num,
            "true_num": true_num}


if __name__ == "__main__":
    batch_num = 10

    epoch = 10
    for e in range(epoch):
        for i, batch_data in enumerate(data_iter.train_iter(batch_num)):
            loss_value = train_step(batch_data["char_id"], batch_data["event_label"],
                                    batch_data["rd_event_type"], batch_data["rd_argument"],
                                    batch_data["rd_argument_mask"])

            if i % 100 == 0:
                logger.info("epoch {0} batch {1} loss value {2}".format(e, i, loss_value))
                evaluation(batch_data, model)

    final_eval = {"hit_num": 0.0, "true_num": 0.0, "pred_num": 0.0}
    for batch_dev in data_iter.dev_iter(batch_num):
        b_eval = evaluation(batch_dev, model)
        for k, v in b_eval.items():
            final_eval[k] += v
    print(eval_metrix(final_eval["hit_num"], final_eval["true_num"], final_eval["pred_num"]))
