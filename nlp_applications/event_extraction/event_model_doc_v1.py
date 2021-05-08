#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    文档级别的事件抽取模型
    ╮(╯▽╰)╭，还是先用pipeline 的结构吧， joint model 实在有点难实现， -_-||
    1）第一步提取所有的实体和事件类型
    2）第二步对
"""

import tensorflow as tf
from nlp_applications.data_loader import LoaderBaiduDueeFin, EventDocument, Event, Argument
from nlp_applications.ner.evaluation import extract_entity

sample_path = "D:\data\\篇章级事件抽取\\"
bd_data_loader = LoaderBaiduDueeFin(sample_path)
max_len = 256
print(bd_data_loader.event2id)
print(bd_data_loader.argument_role2id)

entity_class_num = len(bd_data_loader.argument_role2id)
event_class_num = len(bd_data_loader.event2id)
event2argument = bd_data_loader.event2argument
max_argument_len = max([len(v) for _, v in event2argument.items()])

# for doc in bd_data_loader.document:
#     print(doc.title, len(doc.event_list))
#     # if len(doc.event_list)==3:
#     #     print(doc.text)
#     #     for event in doc.event_list:
#     #         print(bd_data_loader.id2event[event.id])
#     #         for arg in event.arguments:
#     #             print(arg.argument, arg.role)


def cut_sentence(input_sentence):
    innser_sentence_list = []
    sentence_len = len(input_sentence)
    cut_char = {"，", " ", "；", "》", "）", "、", ";"}
    indx = 0

    while indx < sentence_len:

        last_ind = min(indx+max_len, sentence_len)
        if last_ind != sentence_len:
            while last_ind > indx:
                if input_sentence[last_ind-1] in cut_char:
                    break
                last_ind -= 1
        if indx == last_ind:
            print(input_sentence)
            raise Exception
        pre_cut = input_sentence[indx:last_ind]
        innser_sentence_list.append(pre_cut)
        indx = last_ind
    for sentence in innser_sentence_list:
        assert len(sentence) <= max_len
    return innser_sentence_list



class DataIter(object):

    def __init__(self, input_loader, input_batch_num):
        self.input_loader = input_loader
        self.input_batch_num = input_batch_num
        self.entity_label2id = {"O": 0}
        self.max_len = 0

        for e_label in self.input_loader.argument_role2id:
            if e_label == "$unk$":
                continue
            self.entity_label2id[e_label+"_B"] = len(self.entity_label2id)
            self.entity_label2id[e_label + "_I"] = len(self.entity_label2id)

        self.id2entity_label = {v: k for k, v in self.entity_label2id.items()}

    def _search_index(self, target_word, input_sentence_list):
        out_index = (-1, -1)
        for i, sentence in enumerate(input_sentence_list):
            try:
                index_j = sentence.index(target_word)
                out_index = (i, index_j)
                break
            except ValueError as ve:
                pass
                # print(f'Error Message = {ve}')
        if out_index[0] == -1:
            print(input_sentence_list, target_word)
            raise ValueError

        return out_index

    def _transformer2feature(self, input_doc: EventDocument):
        text = input_doc.text
        title = input_doc.title
        sentences = [title]
        sentences_id = []
        split_char = {"。", "\n"}
        sentence = ""
        for char in text:
            if char in split_char:
                sentence += char
                sentence = sentence.strip()
                if len(sentence) > max_len:
                    tiny_sentence_list = cut_sentence(sentence)
                    sentences += tiny_sentence_list
                elif sentence:
                    sentences.append(sentence)
                sentence = ""
            else:
                sentence += char
        sentence = sentence.strip()
        if sentence:
            if len(sentence) > max_len:
                sentences += cut_sentence(sentence)
            else:
                sentences.append(sentence)
        entity_list = set()
        entity_loc_map = dict()
        event_label = [0]*event_class_num
        entity_dict = {(0, 0, 0): 0}
        event_type_list = []
        event_argument_list = []
        event_is_valid = []
        for event in input_doc.event_list:
            event_type_list.append(event.id)
            event_is_valid.append(1)
            sub_event_arg = [0]*max_argument_len
            event_label[event.id] = 1
            for arg in event.arguments:
                if arg.is_enum:
                    continue

                row_ind, column_ind_start = self._search_index(arg.argument, sentences)
                entity_list.add((row_ind, column_ind_start, column_ind_start+len(arg.argument), arg.role))
                entity_loc_map[(row_ind, column_ind_start)] = self.entity_label2id[arg.role+"_B"]
                for ind in range(column_ind_start+1, column_ind_start+len(arg.argument)):
                    entity_loc_map[(row_ind, ind)] = self.entity_label2id[arg.role+"_I"]

                arg_type_id = event2argument[event.id][arg.role_id]
                role_key = (row_ind, column_ind_start, len(arg.argument))
                if role_key not in entity_dict:
                    entity_dict[role_key] = len(entity_dict)
                sub_event_arg[arg_type_id] = entity_dict[role_key]
            event_argument_list.append(sub_event_arg)
        entity_mask_list = []
        entity_loc = []

        id2entity_dict = {v: k for k, v in entity_dict.items()}
        for i in range(len(id2entity_dict)):
            row_id = id2entity_dict[i][0]
            entity_loc.append(row_id)
            row_value = [0]*len(sentences[row_id])
            start_i = id2entity_dict[i][1]
            end_i = id2entity_dict[i][2]
            for iv in range(start_i, end_i):
                row_value[iv] = 1
            entity_mask_list.append(row_value)

        entity_labels = []
        for row_ind, sentence in enumerate(sentences):
            sentence_id = [self.input_loader.char2id[char] for char in sentence]
            entity_label = [entity_loc_map.get((row_ind, col_id), 0) for col_id, char in enumerate(sentence)]

            self.max_len = max(self.max_len, len(sentence_id))
            sentences_id.append(sentence_id)
            entity_labels.append(entity_label)

        return {
            "sentences_id": sentences_id,
            "entity_label": entity_labels,
            "event_label": event_label,
            "entity_mask": entity_mask_list,
            "entity_loc": entity_loc,
            "event_type": event_type_list,
            "event_argument": event_argument_list,
            "event_is_valid": event_is_valid
        }

    def batch_transformer(self, input_batch_data):
        batch_sentences_id = []
        batch_entity_label_id = []
        batch_event_label = []
        batch_entity_mask = []
        batch_entity_loc = []
        batch_event_id = []
        batch_event_argument = []
        batch_event_valid = []

        i_max_len = 0
        max_sentence_num = 0
        max_entity_num = 0
        max_event_num = 0
        for data in input_batch_data:
            max_sentence_num = max(len(data["sentences_id"]), max_sentence_num)
            for sentence_id in data["sentences_id"]:
                i_max_len = max(len(sentence_id), i_max_len)
            batch_event_label.append(data["event_label"])
            max_entity_num = max(len(data["entity_mask"]), max_entity_num)
            max_event_num = max(len(data["event_argument"]), max_event_num)
            batch_event_id.append(data["event_type"])

        for data in input_batch_data:
            sub_sentences_id = data["sentences_id"]
            sub_entity_label_id = data["entity_label"]
            sub_entity_mask = data["entity_mask"]
            sub_entity_loc = data["entity_loc"]
            sub_event_argument = data["event_argument"]
            sub_event_is_valid = data["event_is_valid"]
            sub_len = len(sub_sentences_id)
            if sub_len < max_sentence_num:
                sub_sentences_id += [tf.zeros(i_max_len) for _ in range(max_sentence_num-sub_len)]
                sub_entity_label_id += [tf.zeros(i_max_len) for _ in range(max_sentence_num-sub_len)]

            sub_entity_mask += [tf.zeros(i_max_len) for _ in range(max_entity_num-len(sub_entity_mask))]
            sub_entity_loc += [0 for _ in range(max_entity_num-len(sub_entity_loc))]
            sub_event_argument += [tf.zeros(max_event_num) for _ in range(max_event_num-len(sub_event_argument))]
            sub_event_is_valid += [0 for _ in range(max_event_num-len(sub_event_is_valid))]

            sub_sentences_id = tf.keras.preprocessing.sequence.pad_sequences(sub_sentences_id, padding="post", maxlen=max_len)
            sub_entity_label_id = tf.keras.preprocessing.sequence.pad_sequences(sub_entity_label_id, padding="post", maxlen=max_len)
            sub_entity_mask = tf.keras.preprocessing.sequence.pad_sequences(sub_entity_mask, padding="post", maxlen=max_len)
            sub_event_argument = tf.keras.preprocessing.sequence.pad_sequences(sub_event_argument, padding="post", maxlen=max_argument_len)
            batch_sentences_id.append(sub_sentences_id)
            batch_entity_label_id.append(sub_entity_label_id)
            batch_entity_mask.append(sub_entity_mask)
            batch_entity_loc.append(sub_entity_loc)
            batch_event_argument.append(sub_event_argument)
            batch_event_valid.append(sub_event_is_valid)

        batch_event_id = tf.keras.preprocessing.sequence.pad_sequences(batch_event_id, padding="post")
        return {
            "sentences_id": tf.cast(batch_sentences_id, dtype=tf.int64),
            "entity_labels": tf.cast(batch_entity_label_id, dtype=tf.int64),
            "event_label": tf.cast(batch_event_label, dtype=tf.int64),
            "entity_mask": tf.cast(batch_entity_mask, dtype=tf.float32),
            "entity_loc": tf.cast(batch_entity_loc, dtype=tf.int64),
            "event_id": batch_event_id,
            "event_argument": tf.cast(batch_event_argument, dtype=tf.int64),
            "event_is_valid": tf.cast(batch_event_valid, dtype=tf.int64)
        }

    def __iter__(self):
        inner_batch_data = []
        for doc in self.input_loader.document:
            tf_data = self._transformer2feature(doc)
            if len(tf_data["event_type"]) == 0:
                continue
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        # if inner_batch_data:
        #     yield self.batch_transformer(inner_batch_data)


data_iter = DataIter(bd_data_loader, 2)
entity_class_num = len(data_iter.entity_label2id)
id2entity_label = data_iter.id2entity_label


print(data_iter.max_len, "hello")

vocab_size = len(bd_data_loader.char2id)
embed_size = 64
lstm_size = 64


class EventTree(tf.keras.layers.Layer):

    def __init__(self, arguments_num=0):
        super(EventTree, self).__init__()
        self.event_trigger = tf.keras.layers.Dense(1)
        self.event_arguments = [tf.keras.layers.Dense(1) for _ in range(arguments_num)]

    def call(self, inputs, training=None, mask=None):
        return self.event_trigger(inputs)


class EventModelDocV1(tf.keras.Model):

    def __init__(self):
        super(EventModelDocV1, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.entity_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size, return_sequences=True))
        self.entity_output = tf.keras.layers.Dense(entity_class_num, activation="softmax")
        self.event_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_size))
        self.event_classifier = tf.keras.layers.Dense(event_class_num, activation="sigmoid")
        self.event_embed = tf.keras.layers.Embedding(event_class_num, embed_size)
        self.event_is_valid = tf.keras.layers.Dense(1, activation="sigmoid")
        # self.event_root = [EventTree(len(event2argument[i])) for i in range(event_class_num)]

    def call(self, inputs, entity_mask=None, entity_loc=None, event_id=None, event_argument=None, training=None, mask=None):
        i_batch_num = inputs.shape[0]
        input_id = self.embed(inputs)

        sentence_maxpool = tf.reduce_max(input_id, axis=1)
        sentence_feature = self.event_lstm_layer(sentence_maxpool)
        sentence_feature = tf.reshape(sentence_feature, (i_batch_num, -1))
        sentence_entity_feature = tf.map_fn(lambda x: self.entity_lstm_layer(x), input_id, dtype=tf.float32)
        sentence_entity_label = self.entity_output(sentence_entity_feature)
        event_label = self.event_classifier(sentence_feature)

        event_embed = self.event_embed(event_id)

        batch_event_argument_feature = []
        for batch_id in range(i_batch_num):
            sentence_value = input_id[batch_id]
            entity_mask_value = entity_mask[batch_id]
            entity_mask_value = tf.expand_dims(entity_mask_value, axis=-1)
            entity_loc_value = entity_loc[batch_id]
            event_argument_value = event_argument[batch_id]

            # entity_loc_value = tf.where(entity_loc_value != -1,entity_loc_value)
            entity_loc_sentence = tf.gather(sentence_value, entity_loc_value)

            entity_feature = tf.multiply(entity_loc_sentence, entity_mask_value)
            entity_feature = tf.reduce_max(entity_feature, axis=1)
            event_argument_feature = tf.map_fn(lambda x: tf.gather(entity_feature, x), event_argument_value,
                                               dtype=tf.float32)
            event_argument_feature_shape = event_argument_feature.shape
            if event_argument_feature_shape[0] is None:
                event_argument_feature = tf.reshape(event_argument_feature, (-1, event_argument_feature_shape[1]*event_argument_feature_shape[2]))
            else:
                event_argument_feature = tf.reshape(event_argument_feature, (event_argument_feature_shape[0], -1))
            batch_event_argument_feature.append(event_argument_feature)
        batch_event_argument_feature = tf.stack(batch_event_argument_feature)
        event_feature = tf.concat([event_embed, batch_event_argument_feature], axis=2)
        event_valid_trigger = self.event_is_valid(event_feature)

        return sentence_entity_label, event_label, event_valid_trigger

    def predict(self,
              inputs,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):

        i_batch_num = inputs.shape[0]
        input_id = self.embed(inputs)

        sentence_maxpool = tf.reduce_max(input_id, axis=1)
        sentence_feature = self.event_lstm_layer(sentence_maxpool)
        sentence_feature = tf.reshape(sentence_feature, (i_batch_num, -1))
        sentence_entity_feature = tf.map_fn(lambda x: self.entity_lstm_layer(x), input_id, dtype=tf.float32)
        sentence_entity_label = self.entity_output(sentence_entity_feature)
        event_label = self.event_classifier(sentence_feature)

        batch_res = list()
        for b in range(i_batch_num):
            sentence_entity_label = tf.argmax(sentence_entity_label, axis=-1)
            for sentence_inner_label in sentence_entity_label:
                sentence_inner_label = sentence_inner_label.numpy()



emdv1 = EventModelDocV1()
optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_func2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_func3 = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function(experimental_relax_shapes=True)
def train_step(input_sentences, input_entity_label, input_event_type, input_entity_mask, input_entity_loc,
               input_event_id, input_event_argument, input_event_is_valid):
    with tf.GradientTape() as tape:
        entity_logits, event_logits, event_valid_logits = emdv1(input_sentences, input_entity_mask, input_entity_loc, input_event_id, input_event_argument)
        lossv1 = loss_func(input_entity_label, entity_logits)
        lossv2 = loss_func2(input_event_type, event_logits)
        lossv3 = loss_func3(input_event_is_valid, event_valid_logits)
        lossv = lossv1 + lossv2 + lossv3

    variables = emdv1.variables
    gradients = tape.gradient(lossv, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return lossv


model_path = "D:\\tmp\event_model_doc_v1\\model"
epoch = 10
for e in range(epoch):

    for batch_i, batch_data in enumerate(data_iter):
        loss_value = train_step(batch_data["sentences_id"], batch_data["entity_labels"], batch_data["event_label"],
                                batch_data["entity_mask"], batch_data["entity_loc"], batch_data["event_id"],
                                batch_data["event_argument"], batch_data["event_is_valid"])

        if batch_i % 100 == 0:
            print("epoch {0} batch {1} loss is {2}".format(e, batch_i, loss_value))
            emdv1.save_weights(model_path, save_format='tf')
    break

