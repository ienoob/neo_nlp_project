#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/26 16:59
    @Author  : jack.li
    @Site    : 
    @File    : dueev1_pipeline_model_v1.py

"""
import os
import numpy as np
import tensorflow as tf
from nlp_applications.event_extraction.evaluation import extract_entity, eval_metrix
from nlp_applications.data_loader import LoaderBaiduDueeV1, EventDocument, Event, Argument, BaseDataIterator
from nlp_applications.event_extraction.pipeline_model_v1 import EventTypeClassifier, EventArgument
from nlp_applications.ner.crf_model import CRFNerModel, sent2features, sent2labels

sample_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\句子级事件抽取\\"
bd_data_loader = LoaderBaiduDueeV1(sample_path)
event2argument_dict = bd_data_loader.event2argument
event_num = len(bd_data_loader.event2id)
id2event = bd_data_loader.id2event


class DataIter(BaseDataIterator):

    def __init__(self, data_loader):
        super(DataIter, self).__init__(data_loader)
        self.trigger_bio2id = {
            "O": 0
        }
        for en in range(event_num):
            self.trigger_bio2id["B-{}".format(en)] = len(self.trigger_bio2id)
            self.trigger_bio2id["I-{}".format(en)] = len(self.trigger_bio2id)

        self.event2argument_bio = dict()
        for k, argument in event2argument_dict.items():
            argument_d = {"O": 0}
            for arg in argument:
                argument_d["B-" + arg] = len(argument_d)
                argument_d["I-" + arg] = len(argument_d)
            self.event2argument_bio[k] = argument_d

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

    def single_dev_doc_processor(self, doc: EventDocument):
        text_id = doc.text_id
        raw = doc.text

        event_list = []
        for event in doc.event_list:
            e_id = event.id
            # argument_role2id = self.event2argument_bio[e_id]
            rd_argument = []
            for arg in event.arguments:
                rd_argument.append((arg.start, arg.start+len(arg.argument), arg.role))
            event_value = (e_id, rd_argument)
            event_list.append(event_value)

        return {
            "raw": raw,
            "char_id": tf.cast([text_id], dtype=tf.int32),
            "event_list": event_list
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

    def dev_iter(self):
        for doc in self.data_loader.dev_documents:
            yield self.single_dev_doc_processor(doc)


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
            "char_id": text_id,
            "rd_event_type": [rd_event_type],
            "rd_argument": rd_argument
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_id = []
        batch_rd_event_type = []
        batch_rd_argument = []

        for data in input_batch_data:
            batch_char_id.append(data["char_id"])
            batch_rd_event_type.append(data["rd_event_type"])
            batch_rd_argument.append(data["rd_argument"])
        batch_rd_event_type = tf.cast(batch_rd_event_type, tf.int32)
        batch_rd_argument = tf.keras.preprocessing.sequence.pad_sequences(batch_rd_argument, padding="post")
        batch_char_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_id, padding="post")

        return {
            "char_id": batch_char_id,
            "rd_event_type": batch_rd_event_type,
            "rd_argument": batch_rd_argument
        }

    def train_iter(self, input_batch_num, event_type):
        c_batch_data = []
        for doc in self.data_loader.documents:
            doc_res = self.single_doc_processor(doc, event_type)
            if len(doc_res) == 0:
                continue
            c_batch_data.append(doc_res)
            if len(c_batch_data) == input_batch_num:
                yield self.padding_batch_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_batch_data(c_batch_data)

    def argument_value(self, event_type, data="dev"):
        argument_text = []
        argument_label = []

        if data is "dev":
            for doc in self.data_loader.dev_documents:
                text = doc.text
                argument = ["O" for _ in text]
                for event in doc.event_list:
                    if event != event_type:
                        continue
                    for arg in event.arguments:
                        argument[arg.start] = "B-" + arg.role
                        for iv in range(arg.start + 1, arg.start + len(arg.argument) - 1):
                            argument[iv] = "I-" + arg.role
                argument_text.append(text)
                argument_label.append(argument)

        return argument_text, argument_label


char_size = len(bd_data_loader.char2id)
char_embed = 64
lstm_size = 64
event_class = len(bd_data_loader.event2id)
event_embed_size = 64

data_iter = DataIter(bd_data_loader)
arg_data_iter = DataIterArgument(bd_data_loader)

trigger_num = len(data_iter.trigger_bio2id)
event_model = EventTypeClassifier(char_size, char_embed, lstm_size, event_class, trigger_num)
argument_model_list = [EventArgument(char_size, char_embed, lstm_size, len(arg_data_iter.event2argument_bio[e]))
                       for e in range(1, event_num)]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)


def loss_func1(true_event, pred_event):
    binary_loss = tf.keras.losses.BinaryCrossentropy()
    true_weight = tf.where(tf.greater(true_event, 0), 2.0, 1.0)
    true_event = tf.expand_dims(true_event, axis=-1)
    pred_event = tf.expand_dims(pred_event, axis=-1)
    return binary_loss(true_event, pred_event, true_weight)


def loss_func2(true_trigger, logits_trigger, mask=None):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy()

    return cross_func(true_trigger, logits_trigger, sample_weight=mask)


@tf.function(experimental_relax_shapes=True)
def train_step(input_x, input_event, input_trigger):
    with tf.GradientTape() as tape:
        trigger_logits, event_logits, mask = event_model(input_x)

        lossv = loss_func1(input_event, event_logits) + loss_func2(input_trigger, trigger_logits, mask)

    variable = event_model.trainable_variables
    gradients = tape.gradient(lossv, variable)
    optimizer.apply_gradients(zip(gradients, variable))

    return lossv


epoch = 20
batch_num = 32
model_path = "D:\\tmp\\pipeline_model_v1\\event_model\\model"
# for e in range(epoch):
#
#     for bi, batch_data in enumerate(data_iter.train_iter(batch_num)):
#         loss_value = train_step(batch_data["char_id"], batch_data["event_label"], batch_data["event_trigger"])
#
#         if bi % 100 == 0:
#             print("epoch {0} batch {1} loss value {2}".format(e, bi, loss_value))
#         event_model.save_weights(model_path, save_format='tf')


def loss_func3(true_trigger, logits_trigger, mask=None):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy()

    return cross_func(true_trigger, logits_trigger, sample_weight=mask)


# for event_i in range(1, event_num):
#     current_model = argument_model_list[event_i-1]
#     optimizer = tf.keras.optimizers.Adam()
#
#     print("argument {} start".format(event_i))
#
#     @tf.function(experimental_relax_shapes=True)
#     def train_step(input_x, input_argument):
#         with tf.GradientTape() as tape:
#             argument_logits, mask = current_model(input_x)
#
#             lossv = loss_func3(input_argument, argument_logits, mask)
#
#         variable = current_model.trainable_variables
#         gradients = tape.gradient(lossv, variable)
#         optimizer.apply_gradients(zip(gradients, variable))
#
#         return lossv
#
#
#     model_path = "D:\\tmp\\pipeline_model_v1\\argument_model_{}".format(event_i)
#     if not os.path.exists(model_path):
#         os.mkdir(model_path)
#     model_path += "\\model"
#
#     for e in range(epoch):
#
#         for bi, batch_data in enumerate(arg_data_iter.train_iter(batch_num, event_i)):
#             loss_value = train_step(batch_data["char_id"], batch_data["rd_argument"])
#
#             # if bi % 100 == 0:
#             print("model {0} epoch {1} batch {2} loss value {3}".format(event_i, e, bi, loss_value))
#             current_model.save_weights(model_path, save_format='tf')

def crf_train():
    for event_i in range(1, event_num):
        print("crf model {} start".format(event_i))
        model_path = "D:\\tmp\\pipeline_model_v1\\crf_{}.model".format(event_i)
        crf_model = CRFNerModel(is_save=True)
        crf_model.save_model = model_path

        train_sentence, train_label = arg_data_iter.argument_value(event_i)
        X_train = [sent2features(s) for s in train_sentence]
        y_train = [sent2labels(s) for s in train_label]

        crf_model.fit(X_train, y_train)


# crf_train()

def predict():
    use_crf_model = True
    event_model.load_weights(model_path)
    hit_num = 0.0
    real_num = 0.0
    predict_num = 0.0
    event_hit_num = 0.0
    trigger_bio2id = data_iter.trigger_bio2id
    trigger_id2bio = {v:k for k, v in trigger_bio2id.items()}
    for i, data in enumerate(data_iter.dev_iter()):
        char_id = data["char_id"]
        event_list = data["event_list"]
        trigger_logits, event_logits, mask = event_model(char_id)

        trigger_pred = tf.argmax(trigger_logits, axis=-1).numpy()
        trigger_bio = [trigger_id2bio[t] for t in trigger_pred[0]]
        trigger_bio_pred = [int(ee[2]) for ee in extract_entity(trigger_bio)]
        event_pred = tf.greater(event_logits, 0.5).numpy()
        real_num += len(event_list)

        event_dict = dict()
        for eve in event_list:
            event_dict.setdefault(eve[0], [])
            event_dict[eve[0]].append(eve[1])
        print(data["raw"])
        print(event_list)
        predict_list = []
        # for ei, ev in enumerate(event_pred[0]):
        #     if not ev:
        #         continue
        #     if ei == 0:
        #         continue
        for ei in trigger_bio_pred:
            # print(ei)
            # print(id2event[ei])
            if use_crf_model:
                crf_md = CRFNerModel()
                crf_md.save_model = "D:\\tmp\\pipeline_model_v1\\crf_{}.model".format(ei)
                crf_md.load_model()

                X_train = [sent2features(s) for s in [data["raw"]]]
                argument_pred_bio = crf_md.predict_list(X_train)
                arg_extract = extract_entity(argument_pred_bio[0])
            else:
                argument_model_path = "D:\\tmp\\pipeline_model_v1\\argument_model_{}\\model".format(ei)
                argument_model = argument_model_list[ei-1]
                argument_model.load_weights(argument_model_path)

                argument_logits, _ = argument_model(char_id)
                argument_pred = tf.argmax(argument_logits, axis=-1).numpy()

                argument_role2id = data_iter.event2argument_bio[ei]
                argument_id2role = {v:k for k, v in argument_role2id.items()}
                argument_pred_bio = [argument_id2role[a] for a in argument_pred[0]]
                arg_extract = extract_entity(argument_pred_bio)

            if ei in event_dict:
                event_hit_num += 1
                if arg_extract in event_dict[ei]:
                    hit_num += 1
            # print(ei, extract_entity(argument_pred_bio))
            predict_list.append((ei, arg_extract))

            predict_num += 1
        print(predict_list)
        # print(data["event_list"])
    print(hit_num, real_num, predict_num, event_hit_num)
    print(eval_metrix(hit_num, real_num, predict_num))


if __name__ == "__main__":
    predict()
