#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import json
import logging
import tensorflow as tf
from nlp_applications.data_loader import load_json_line_data
from nlp_applications.utils import load_word_vector
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertMainLayer
from nlp_applications.ner.crf_addons import crf_log_likelihood, viterbi_decode
from nlp_applications.ner.evaluation import extract_entity, hard_score_res_v2

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

schema_path = "D:\data\句子级事件抽取\duee_schema\\duee_event_schema.json"
data_path = "D:\data\句子级事件抽取\duee_train.json\\duee_train.json"
eval_data_path = "D:\data\句子级事件抽取\duee_dev.json\\duee_dev.json"
word_embed_path = "D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char"

schema_data = load_json_line_data(schema_path)
train_data = load_json_line_data(data_path)
eval_data = load_json_line_data(eval_data_path)

train_data = list(train_data)
eval_data = list(eval_data)


class BertCrfModel(tf.keras.Model):

    def __init__(self, inner_bert_model_name, class_num):
        super(BertCrfModel, self).__init__()
        # config = BertConfig.from_pretrained(inner_bert_model_name, cache_dir=None)
        self.bert_model = TFBertModel.from_pretrained(inner_bert_model_name)
        # self.bert_model = TFBertModel(config)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(class_num, class_num)))
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs, training=None, mask=None, labels=None):
        # seg_id = tf.zeros(mask.shape)
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        outputs = self.bert_model(inputs, attention_mask=mask)
        outputs = outputs[0]
        outputs = self.drop_out(outputs, training)
        out_tags = self.out(outputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = crf_log_likelihood(out_tags, label_sequences, text_lens, self.transition_params)

            return out_tags, text_lens, log_likelihood
        else:
            return out_tags, text_lens




class ATTBertModel(object):

    def __init__(self, bert_model_name, class_num):
        self.model = BertCrfModel(bert_model_name, class_num)

    def fit(self, train_data, label_data, mask_data):
        optimizer = tf.keras.optimizers.Adam(1e-2)

        def loss_func(input_y, logits):
            cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            mask = tf.math.logical_not(tf.math.equal(input_y, 0))

            mask = tf.cast(mask, dtype=tf.int64)
            lossv = cross_func(input_y, logits, sample_weight=mask)

            return lossv

        @tf.function()
        def train_step(input_xx, input_yy, input_mask):
            with tf.GradientTape() as tape:
                logits, _, log_likelihood = self.model(input_xx, True, input_mask, input_yy)
                loss_v1 = -tf.reduce_mean(log_likelihood)
                loss_v2 = loss_func(input_yy, logits)
                loss_v = loss_v1 + loss_v2

            variables = self.model.trainable_variables
            gradients = tape.gradient(loss_v, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return loss_v

        epoch = 200
        for ep in range(epoch):
            loss = train_step(train_data, label_data, mask_data)

            if epoch % 10 == 0:
                print("loss value is {}".format(loss))

            # for batch_i, (sentence_id, label_id, sentence_mask) in enumerate(input_dataset.take(-1)):
            #     loss = train_step(sentence_id, label_id, sentence_mask)
            #
            #     if batch_i % 10 == 0:
            #         print("epoch {0} batch {1} loss is {2}".format(ep, batch_i, loss))

    def predict(self, train_data, mask_data):

        out_tags, text_lens = self.model(train_data, mask=mask_data)

        return out_tags, text_lens

    @property
    def transition_params(self):
        return self.model.transition_params





def sample_data(input_event_type, input_event_role, data_source):
    train_data_list = []
    text_list = []
    label_list = []
    for data in data_source:
        arguments = set()
        for event in data["event_list"]:
            if event["event_type"] != input_event_type:
                continue
            for arg in event["arguments"]:
                if arg["role"] != input_event_role:
                    continue
                train_data_list.append((data["text"], arg["argument_start_index"], arg["argument"]))
                arguments.add((arg["argument_start_index"], arg["argument"]))
        if arguments:
            text_list.append(data["text"])
            label_list.append(arguments)

    return train_data_list, text_list, label_list


bert_model_name = "bert-base-chinese"
def generate(input_data, input_label):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    label2id = {
        "pad": 0,
        "B-E": 1,
        "I-E": 2,
        "O": 3
    }
    sentence_list = []
    label_list = []
    sentence_mask_list = []
    maxlen_value = 0
    for i, text_data in enumerate(input_data):
        label_value_list = input_label[i]
        loc_d = dict()
        for start, span in label_value_list:
            loc_d[start] = "B-E"
            for iv in range(start+1, start+len(span)):
                loc_d[iv] = "I-E"
        sentence_id = tokenizer.encode(text_data)
        label_value = ["O"]+[loc_d.get(s_i, "O") for s_i, _ in enumerate(sentence_id)]+["O"]
        label_id = [label2id[lv] for lv in label_value]
        sentence_mask = [1 for _ in sentence_id]

        sentence_list.append(sentence_id)
        label_list.append(label_id)
        sentence_mask_list.append(sentence_mask)
        maxlen_value = max(maxlen_value, len(sentence_id))

    train_data = tf.keras.preprocessing.sequence.pad_sequences(sentence_list, padding="post", maxlen=maxlen_value)
    label_data = tf.keras.preprocessing.sequence.pad_sequences(label_list, padding="post", maxlen=maxlen_value)
    mask_data = tf.keras.preprocessing.sequence.pad_sequences(sentence_mask_list, padding="post", maxlen=maxlen_value)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, label_data, mask_data)).shuffle(10).batch(10)

    return dataset, (train_data, label_data, mask_data)




for schema in schema_data:
    event_type = schema["event_type"]
    for role in schema["role_list"]:
        role_value = role["role"]

        test_train, test_train_data, test_train_label = sample_data(event_type, role_value, train_data)
        test_eval, test_eval_data, test_eval_label = sample_data(event_type, role_value, eval_data)

        logger.info("event {0} start, role {1} start, train_data {2}".format(event_type, role_value, len(test_train)))
        if len(test_train) == 0:
            continue

        test_train_dataset, (train_data_t, label_data_t, mask_data_t) = generate(test_train_data, test_train_label)

        att_model = ATTBertModel(bert_model_name, 4)
        att_model.fit(train_data_t, label_data_t, mask_data_t)

        # _, (train_data_v, label_data_v, mask_data_v) = generate(test_eval_data, test_eval_label)
        logits, text_lens = att_model.predict(train_data_t, mask_data_t)
        print(label_data_t)
        paths = []
        for logit, text_len in zip(logits, text_lens):
            viterbi_path, _ = viterbi_decode(logit[:text_len], att_model.transition_params)
            paths.append(viterbi_path)
        id2label = {
            0: "O",
            1: "B-E",
            2: "I-E",
            3: "O"}
        paths2label = [[id2label[p] for p in path] for path in paths]
        extract_info = [[(ee[0], test_train_data[i][ee[0]:ee[1]]) for ee in extract_entity(path)] for i, path in enumerate(paths2label)]
        # extract_info1 = [[(ee[0], test_train_data[i][ee[0]:ee[1]]) for ee in extract_entity(path)] for i, path in
        #                 enumerate(paths2label)]
        print(extract_info)

        print(hard_score_res_v2(test_train_label, extract_info))


    break
