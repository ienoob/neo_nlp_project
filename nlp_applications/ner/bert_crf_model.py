#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

"""
    使用transformer+tensorflow2 实现 bert + crf
"""
import time
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertMainLayer, TFBertPreTrainedModel
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.crf_addons import crf_log_likelihood, viterbi_decode
from nlp_applications.ner.evaluation import metrix

msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")
bert_model_name = "bert-base-chinese"
class_num = len(msra_data.label2id)
lstm_dim = 64


class DataIterator(object):
    def __init__(self, input_loader, input_batch_num):
        self.input_loader = input_loader
        self.input_batch_num = input_batch_num
        # self.entity_label2id = {"O": 1, "pad": 0}
        self.max_len = 0
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def _transformer2feature(self, sentence, label_list):
        if len(sentence) > 500:
            sentence = sentence[:500]
            label_list = label_list[:500]
        sentence_id = self.tokenizer.encode(sentence)
        label_id = [self.input_loader.label2id[label_i] for label_i in label_list]
        label_id = [0] + label_id + [0]
        # assert len(sentence_id) == len(label_id)
        sentence_mask = [1 for _ in sentence_id]
        # sentence_mask[0] = 0
        # sentence_mask[-1] = 0

        return {
            "sentence_id": sentence_id,
            "label_id": label_id,
            "sentence_mask": sentence_mask,
            "sentence": sentence
        }

    def batch_transformer(self, input_batch_data):
        batch_sentence_id = []
        batch_label_id = []
        batch_sentence_mask = []
        batch_sentence = []
        for data in input_batch_data:
            batch_sentence_id.append(data["sentence_id"])
            batch_label_id.append(data["label_id"])
            batch_sentence_mask.append(data["sentence_mask"])
            batch_sentence.append(data["sentence"])

        batch_sentence_id = tf.keras.preprocessing.sequence.pad_sequences(batch_sentence_id, padding="post", maxlen=512)
        batch_label_id = tf.keras.preprocessing.sequence.pad_sequences(batch_label_id, padding="post", maxlen=512)
        batch_sentence_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_sentence_mask, padding="post", maxlen=512)
        return {
            "sentence_id": batch_sentence_id,
            "label_id": batch_label_id,
            "sentence_mask": batch_sentence_mask,
            "sentence": batch_sentence
        }

    def __iter__(self):
        inner_batch_data = []
        for i, sentence in enumerate(self.input_loader.train_sentence_list):
            tf_data = self._transformer2feature(sentence, self.input_loader.train_tag_list[i])
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        if inner_batch_data:
            yield self.batch_transformer(inner_batch_data)

    def dev_iter(self):
        inner_batch_data = []
        for i, sentence in enumerate(self.input_loader.test_sentence_list):
            tf_data = self._transformer2feature(sentence, self.input_loader.test_tag_list[i])
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        if inner_batch_data:
            yield self.batch_transformer(inner_batch_data)


class BertCrfModel(tf.keras.Model):

    def __init__(self, inner_bert_model_name):
        super(BertCrfModel, self).__init__()
        # config = BertConfig.from_pretrained(inner_bert_model_name, cache_dir=None)
        self.bert_model = TFBertModel.from_pretrained(inner_bert_model_name, output_attentions=True, output_hidden_states=True)
        # self.bert_model = TFBertModel(config)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(class_num, class_num)))
        self.out = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs, training=None, mask=None, labels=None):

        mask = tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32)
        seg_id = tf.cast(tf.zeros(mask.shape), dtype=tf.int32)
        text_lens = tf.math.reduce_sum(mask, axis=-1)
        outputs = self.bert_model(inputs, mask, seg_id)
        # print(outputs[0].shape, outputs[1].shape)
        outputs = outputs[0]

        out_tags = self.out(outputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = crf_log_likelihood(out_tags, label_sequences, text_lens, self.transition_params)

            return out_tags, text_lens, log_likelihood, mask
        else:
            return out_tags, text_lens


class BertCrfModelV2(tf.keras.Model):

    def __init__(self, inner_bert_model_name):
        super(BertCrfModelV2, self).__init__()
        # config = BertConfig.from_pretrained(inner_bert_model_name, cache_dir=None)
        self.bert_model = TFBertModel.from_pretrained(inner_bert_model_name)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim, return_sequences=True))
        # self.bert_model = TFBertModel(config)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(class_num, class_num)))
        self.out = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs, training=None, mask=None, labels=None):
        # seg_id = tf.zeros(mask.shape)
        mask = tf.math.logical_not(tf.math.equal(inputs, 0))
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1)
        outputs = self.bert_model(inputs)
        outputs = outputs[0]
        inputs = self.bi_lstm(outputs, mask=mask)
        out_tags = self.out(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = crf_log_likelihood(out_tags, label_sequences, text_lens, self.transition_params)

            return out_tags, text_lens, log_likelihood
        else:
            return out_tags, text_lens


config = BertConfig.from_pretrained(bert_model_name)
bert_crf_model = BertCrfModel(bert_model_name)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def loss_func(input_y, logits, mask):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy()
    lossv = cross_func(input_y, logits, sample_weight=mask)

    return lossv

def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = viterbi_decode(logit[:text_len], bert_crf_model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy

def predict():
    final_res = []
    for batch_i, batch in tqdm(enumerate(data_iterator.dev_iter())):
        input_s_list = batch["sentence"]
        logits, text_lens = bert_crf_model(batch["sentence_id"])

        paths = []
        for logit, text_len in zip(logits, text_lens):
            viterbi_path, _ = viterbi_decode(logit[:text_len], bert_crf_model.transition_params)
            paths.append(viterbi_path[1:])

        output_label = []
        for i, output_id in enumerate(paths):
            olen = len(output_id)
            ilen = len(input_s_list[i])
            if olen < ilen:
                output_label.append([msra_data.id2label[o] for o in output_id]+["O"]*(ilen-olen))
            else:
                output_label.append([msra_data.id2label[o] for o in output_id][:ilen])
        # print(len(output_label), batch_i)
        final_res += output_label
        # break
    return final_res

def evaluation():
    start = time.time()
    predict_labels = predict()
    # print(predict_labels[0])

    true_labels = msra_data.test_tag_list
    true_labels = [true_labels[i][:len(plvalue)] for i, plvalue in enumerate(predict_labels)]
    # print(true_labels[0])
    print(metrix(true_labels, predict_labels))
    print("eval cost {}".format(time.time()-start))


@tf.function(experimental_relax_shapes=True)
def train_step(input_xx, input_yy, input_mask):

    with tf.GradientTape() as tape:
        logits, text_len, log_likelihood, m_mask = bert_crf_model(input_xx, True, None, input_yy)
        loss_v1 = -tf.reduce_mean(log_likelihood)
        loss_v2 = loss_func(input_yy, logits, m_mask)
        loss_v = loss_v1 + loss_v2

    variables = bert_crf_model.trainable_variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v, logits, text_len

batch_num = 8
bert_crf_model_path = "D:\\tmp\\bert_crf"

# ckpt = tf.train.Checkpoint(optimizer=optimizer, model=bert_crf_model)
# ckpt.restore(tf.train.latest_checkpoint(bert_crf_model_path))
# ckpt_manager = tf.train.CheckpointManager(ckpt,
#                                           bert_crf_model_path,
#                                           checkpoint_name='model.ckpt',
#                                           max_to_keep=1)


data_iterator = DataIterator(msra_data, batch_num)
epoch = 1
for ep in range(epoch):

    for batch_i, batch in enumerate(data_iterator):
        loss, logits, text_lens = train_step(batch["sentence_id"], batch["label_id"], batch["sentence_mask"])

        if batch_i % 10 == 0:
            accuracy = get_acc_one_step(logits, text_lens, batch["label_id"])
            print("epoch {0} batch {1} loss is {2}, accuracy {3}".format(ep, batch_i, loss, accuracy))
        if batch_i and batch_i % 100 == 0:
            evaluation()
            # ckpt_manager.save()
            # bert_crf_model.save_weights(bert_crf_model_path, save_format='tf')
#
#
# ckpt = tf.train.Checkpoint(optimizer=optimizer,model=bert_crf_model)
# ckpt.restore(tf.train.latest_checkpoint(bert_crf_model_path))

#
# def predict(input_s_list):
#     # max_v_len = max([len(input_s) for input_s in input_s_list])
#     dataset = tf.keras.preprocessing.sequence.pad_sequences([data_iterator.tokenizer.encode(input_str) for input_str in input_s_list], padding='post', maxlen=512)
#     logits, text_lens = bert_crf_model.predict(dataset)
#     paths = []
#     for logit, text_len in zip(logits, text_lens):
#         viterbi_path, _ = viterbi_decode(logit[:text_len], bert_crf_model.transition_params)
#         paths.append(viterbi_path)
#
#     output_label = []
#     for i, output_id in enumerate(paths):
#         olen = len(output_id)
#         ilen = len(input_s_list[i])
#         if olen < ilen:
#             output_label.append([msra_data.id2label[o] for o in output_id]+["O"]*(ilen-olen))
#         else:
#             output_label.append([msra_data.id2label[o] for o in output_id][:ilen])
    # output_label = [[id2label[o] for o in output_id] for i, output_id in
    #                 enumerate(paths)]

    # return output_label







