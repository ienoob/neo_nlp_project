#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import jieba
import numpy as np
import tensorflow as tf
from nlp_applications.ie_relation_extraction.evaluation import eval_metrix
from tensorflow.python.ops import math_ops
from nlp_applications.data_loader import LoaderDuie2Dataset, Document
from nlp_applications.utils import load_word_vector
from nlp_applications.ie_relation_extraction.join.pointer_net import PointerNet

data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取"
# word_embed_path = "D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char"
data_loader = LoaderDuie2Dataset(data_path)
# word_embed = load_word_vector(word_embed_path)
# word_embed = {}

batch_num = 5
vocab_size = len(data_loader.char2id)
word_size = len(data_loader.word2id)
predicate_num = len(data_loader.relation2id)
embed_size = 32
word_embed_size = 32
lstm_size = 64


class DataIterator(object):
    def __init__(self, input_data_loader):
        self.data_loader = input_data_loader
        self.word2id = dict()

    def single_doc_processor(self, doc: Document):
        encoding = doc.text_id
        text_raw = doc.raw_text
        encoding_mask = [1]*len(encoding)
        sub_span = np.zeros(len(encoding))
        sub_label = np.zeros((2, len(encoding)))
        po_label = np.zeros((2*predicate_num, len(encoding)))
        evaluation_label = []
        word_encode_id = []
        for tword in jieba.cut(text_raw):
            word_encode_id += [self.data_loader.word2id.get(tword, 1)] * len(tword)

        relation_random = np.random.choice(doc.relation_list)

        sub_loc = (relation_random.sub.start, relation_random.sub.end-1)
        sub_span[relation_random.sub.start: relation_random.sub.end] = 1
        for relation in doc.relation_list:
            sub = relation.sub
            obj = relation.obj

            evaluation_label.append((sub.start, sub.end - 1, obj.start, obj.end - 1, relation.id))
            sub_label[0][relation.sub.start] = 1
            sub_label[1][relation.sub.end - 1] = 1
            if relation.sub.start != relation_random.sub.start:
                continue
            if relation.sub.end != relation_random.sub.end:
                continue

            pre_type = relation.id
            obj_start = relation.obj.start
            obj_end = relation.obj.end - 1
            po_label[pre_type * 2][obj_start] = 1
            po_label[pre_type * 2 + 1][obj_end] = 1

        return {"encoding": encoding,
                "word_encode_id": word_encode_id,
                "encoding_mask": encoding_mask,
                "sub_span": sub_span,
                "sub_label": sub_label,
                "po_label": po_label,
                "sub_loc": sub_loc,
                "entity_relation_value": evaluation_label}

    def single_test_doc_processor(self, doc: Document):
        encoding = doc.text_id
        text = doc.raw_text
        encoding_mask = [1]*len(encoding)

        return {"encoding": encoding,
                "encoding_mask": encoding_mask,
                }

    def padding_batch_data(self, input_batch_data):
        batch_encoding = []
        batch_word_encode_id = []
        batch_encoding_mask = []
        batch_sub_span = []
        batch_sub_label = []
        batch_po_label = []
        batch_sub_loc = []
        batch_entity_relation_value = []
        max_len = 0
        for data in input_batch_data:
            batch_encoding.append(data["encoding"])
            batch_word_encode_id.append(data["word_encode_id"])
            batch_encoding_mask.append(data["encoding_mask"])
            batch_sub_span.append(data["sub_span"])
            batch_sub_loc.append(data["sub_loc"])
            batch_entity_relation_value.append(data["entity_relation_value"])

            max_len = max(len(data["encoding"]), max_len)

        for data in input_batch_data:
            sub_label = data["sub_label"]
            sub_label = np.pad(sub_label, ((0, 0), (0, max_len-sub_label.shape[1])), 'constant', constant_values=0)
            po_label = data["po_label"]
            po_label = np.pad(po_label, ((0, 0), (0, max_len - po_label.shape[1])), 'constant', constant_values=0)
            batch_sub_label.append(sub_label)
            batch_po_label.append(po_label)

        return {
            "encoding": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding, padding="post"),
            "word_encode_id": tf.keras.preprocessing.sequence.pad_sequences(batch_word_encode_id, padding="post"),
            "encoding_mask": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding_mask, padding="post"),
            "sub_span": tf.keras.preprocessing.sequence.pad_sequences(batch_sub_span, padding="post", dtype="float32"),
            "sub_label": tf.cast(batch_sub_label, dtype=tf.int32),
            "po_label": tf.cast(batch_po_label, dtype=tf.int32),
            "sub_loc": tf.cast(batch_sub_loc, dtype=tf.int32),
            "entity_relation_value": batch_entity_relation_value}

    def padding_test_batch_data(self, input_batch_data):
        batch_encoding = []
        batch_encoding_mask = []
        max_len = 0
        for data in input_batch_data:
            batch_encoding.append(data["encoding"])
            batch_encoding_mask.append(data["encoding_mask"])
            max_len = max(len(data["encoding"]), max_len)
        return {
            "encoding": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding, padding="post"),
            "encoding_mask": tf.keras.preprocessing.sequence.pad_sequences(batch_encoding_mask, padding="post")
        }

    def train_iter(self, input_batch_num):
        batch_data = []
        rg_idxs = np.arange(0, len(self.data_loader.documents))
        np.random.shuffle(rg_idxs)
        for doc_i in rg_idxs:
            doc = self.data_loader.documents[doc_i]
        # for doc in self.data_loader.documents:
            batch_data.append(self.single_doc_processor(doc))
            if len(batch_data) == input_batch_num:
                yield self.padding_batch_data(batch_data)
                batch_data = []
        if batch_data:
            yield self.padding_batch_data(batch_data)

    def test_iter(self, input_batch_num):
        batch_data = []
        for doc in self.data_loader.test_documents:
            batch_data.append(self.single_test_doc_processor(doc))
            if len(batch_data) == input_batch_num:
                yield self.padding_test_batch_data(batch_data)
                batch_data = []
        if batch_data:
            yield self.padding_test_batch_data(batch_data)

    def dev_iter(self, input_batch_num):
        batch_data = []
        for doc in self.data_loader.dev_documents:
            batch_data.append(self.single_doc_processor(doc))
            if len(batch_data) == input_batch_num:
                yield self.padding_batch_data(batch_data)
                batch_data = []
        if batch_data:
            yield self.padding_batch_data(batch_data)


def check_sub(input_sub_value):
    sub_list = []
    for b, s_pred in enumerate(input_sub_value):

        sub = []
        for j, sv in enumerate(s_pred[0]):
            if sv == 0:
                continue
            for k, pv in enumerate(s_pred[1]):
                if k < j:
                    continue
                if pv == 0:
                    continue
                # entity_list.append((j, k))
                sub.append((j, k))
        sub_list.append(sub)
    print("check_sub ", sub_list)


def check_po(input_po_value):
    spo_list = []
    for mi in range(predicate_num):
        po_s_array = input_po_value[mi * 2]
        po_e_array = input_po_value[mi * 2 + 1]

        for mj, pvs in enumerate(po_s_array):
            if pvs == 0:
                continue
            for mk, pve in enumerate(po_e_array):
                if mk < mj:
                    continue
                if pve == 0:
                    continue
                spo_list.append((mj, mk, mi))
    print("check po ", spo_list)


def evaluation(batch_data, input_model):
    # check_sub(batch_data["sub_label"].numpy())
    predict_batch_spo = input_model(batch_data["encoding"], batch_data["word_encode_id"])
    real_batch_spo = batch_data["entity_relation_value"]
    hit_num = 0.0
    predict_num = 0.0
    real_num = 0.0
    predict_wrong_list = []
    recall_fail_list = []
    for b, spo_pred in enumerate(predict_batch_spo):
        # check_po(batch_data["po_label"][b].numpy())
        true_res = real_batch_spo[b]
        real_num += len(true_res)
        predict_wrong_row = []
        hit_index = []

        predict_num += len(spo_pred)
        for ei, ej, pi, pj, pt in spo_pred:
            pre_value = (ei, ej, pi, pj, pt)
            if pre_value in true_res:
                hit_num += 1
                hit_index.append(true_res.index(pre_value))
            else:
                predict_wrong_row.append(pre_value)
        recall_fail_row = [true_res[i] for i in range(len(true_res)) if i not in hit_index]
        predict_wrong_list.append(predict_wrong_row)
        recall_fail_list.append(recall_fail_row)

    return {
        "hit_num": hit_num,
        "predict_num": predict_num,
        "real_num": real_num,
        "predict_wrong": predict_wrong_list,
        "recall_fail": recall_fail_list
    }


def dev_evaluation(data_iter, model):
    test_batch_num = 10
    final_res = {"hit_num": 0.0, "real_num": 0.0, "predict_num": 0.0}
    batch_data_iter = data_iter.dev_iter(test_batch_num)
    for batch_data in batch_data_iter:
        e_res = evaluation(batch_data, model)
        print("eval => {}".format(e_res))
        final_res["hit_num"] += e_res["hit_num"]
        final_res["real_num"] += e_res["real_num"]
        final_res["predict_num"] += e_res["predict_num"]

    eval_res = eval_metrix(final_res["hit_num"], final_res["real_num"], final_res["predict_num"])
    print("evaluation dev res is {}".format(eval_res))


from tf2.python.custom_schedule import CustomSchedule

def main():
    # boundaries = [100000, 110000]
    # values = [0.01, 0.001, 0.001]
    #
    # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    learing_rate = CustomSchedule(128*3, 4000)

    pm_model = PointerNet(vocab_size, embed_size, word_size, word_embed_size, lstm_size, predicate_num)
    data_iter = DataIterator(data_loader)
    #

    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9,
                                    beta_2=0.98, epsilon=1e-9)

    def loss_func(input_y, logits, input_mask):
        mask_logic = tf.math.logical_not(tf.math.equal(input_y, 0))
        mask = tf.where(mask_logic, 10.0, 1.0)
        mask *= tf.cast(input_mask, dtype=tf.float32)
        # loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # input_y = tf.expand_dims(input_y, axis=-1)
        # logits = tf.expand_dims(logits, axis=-1)
        input_y = tf.cast(input_y, dtype=tf.float32)

        input_y *= mask
        logits *= mask
        loss_va = tf.reduce_mean(tf.keras.losses.mse(input_y, logits))
        # loss_va = loss_fun(input_y, logits, sample_weight=mask)

        return loss_va

    def loss_func_v2(input_y, logits, input_mask):
        mask_logic = tf.math.logical_not(tf.math.equal(input_y, 0))
        mask = tf.where(mask_logic, 3.0, 1.0)
        mask *= tf.cast(input_mask, dtype=tf.float32)
        loss_fun = tf.keras.losses.BinaryCrossentropy()
        input_y = tf.expand_dims(input_y, axis=-1)
        logits = tf.expand_dims(logits, axis=-1)

        loss_va = loss_fun(input_y, logits, sample_weight=mask)

        return loss_va

    # loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(experimental_relax_shapes=True)
    def train_step(input_x, input_word_x, input_sub_loc, input_sub_label, input_po_label):
        with tf.GradientTape() as tape:
            sub_logits, po_logits, data_mask = pm_model(input_x, input_word_x, input_sub_loc, training=True)
            data_mask = tf.expand_dims(data_mask, 1)
            sub_data_mask = tf.repeat(data_mask, 2, axis=1)
            po_data_mask = tf.repeat(data_mask, 2*predicate_num, axis=1)
            # print(data_mask)
            # print(sub_data_mask.shape, sub_logits.shape)
            # print(po_data_mask.shape, po_logits.shape)
            # sub_logits = sub_logits * tf.cast(sub_data_mask, dtype=tf.float32)
            # po_logits = po_logits * tf.cast(po_data_mask, dtype=tf.float32)
            lossv = 5*loss_func_v2(input_sub_label, sub_logits, sub_data_mask) + loss_func_v2(input_po_label, po_logits, po_data_mask)
            # print(lossv)
        variables = pm_model.variables
        gradients = tape.gradient(lossv, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return lossv

    epoch = 10
    # model_path = "D:\\tmp\\pointer_net_model\\model"
    # pm_model.load_weights(model_path)

    for ep in range(epoch):
        for batch_i, b_data in enumerate(data_iter.train_iter(batch_num)):
            loss_value = train_step(b_data["encoding"], b_data["word_encode_id"], b_data["sub_loc"],
                                    b_data["sub_label"], b_data["po_label"])

            if batch_i % 100 == 0:
                print("epoch {0} batch {1} loss value is {2}".format(ep, batch_i, loss_value))
                # print(evaluation(b_data, pm_model))
                # pm_model.save_weights(model_path, save_format='tf')
        dev_evaluation(data_iter, pm_model)


    # test_batch_num = 10
    # final_res = {"hit_num": 0.0, "real_num": 0.0, "predict_num": 0.0}
    # batch_data_iter = data_iter.dev_iter(test_batch_num)
    # for batch_data in batch_data_iter:
    #     e_res = evaluation(batch_data, pm_model)
    #     print("eval => {}".format(e_res))
    #     final_res["hit_num"] += e_res["hit_num"]
    #     final_res["real_num"] += e_res["real_num"]
    #     final_res["predict_num"] += e_res["predict_num"]
    #
    # eval_res = eval_metrix(final_res["hit_num"], final_res["real_num"], final_res["predict_num"])
    # print(eval_res)
    # test_batch_num = 1
    # pm_model.load_weights(model_path)
    # batch_data_iter = data_iter.dev_iter(test_batch_num)
    # submit_res = []
    # batch_i = 0
    # save_path = "D:\\tmp\submit_data\\duie.json"
    # for i, batch_data in enumerate(batch_data_iter):
    #     print("batch {} start".format(i))
    #     out_sub_preds, out_po_preds = pm_model.predict(batch_data["encoding"])
    #
    #     print(out_sub_preds, out_po_preds)
    #
    #     break


if __name__ == "__main__":
    main()
