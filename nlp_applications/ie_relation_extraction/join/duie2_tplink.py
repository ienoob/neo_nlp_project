#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/6/4 21:30
    @Author  : jack.li
    @Site    : 
    @File    : duie2_tplink.py

"""
import jieba
import numpy as np
import tensorflow as tf
from nlp_applications.ie_relation_extraction.evaluation import eval_metrix
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from nlp_applications.ie_relation_extraction.join.tplink import Tplink

data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)
triple_regularity = data_loader.triple_set

batch_num = 5
char_size = len(data_loader.char2id)
word_size = len(data_loader.word2id)
char_embed = 32
word_embed = 32
entity_embed_size = 32
rel_num = len(data_loader.relation2id)
rel_embed_size = 32
lstm_size = 64

print(data_loader.max_seq_len)
print("rel_num {}".format(rel_num))


class DataIter(BaseDataIterator):

    def __init__(self, input_loader):
        super(DataIter, self).__init__(input_loader)
        self.data_loader = input_loader

    def single_doc_processor(self, doc: Document):
        char_encode_id = doc.text_id
        text_raw = doc.raw_text

        word_encode_id = []
        for tword in jieba.cut(text_raw):
            word_encode_id += [self.data_loader.word2id.get(tword, 1)] * len(tword)
        assert len(word_encode_id) == len(char_encode_id)
        entity_label_data = np.zeros((len(text_raw), len(text_raw)))
        hh_label_data = np.zeros((len(text_raw), len(text_raw)))
        tt_label_data = np.zeros((len(text_raw), len(text_raw)))
        mt_entity_mask = np.ones((len(text_raw), len(text_raw)))
        mt_entity_mask = np.triu(mt_entity_mask, k=0)
        mt_mask = np.ones((len(text_raw), len(text_raw)))

        entity_relation_value = []
        for relation in doc.relation_list:
            sub = relation.sub
            obj = relation.obj

            entity_label_data[sub.start][sub.end - 1] = 1
            entity_label_data[obj.start][obj.end - 1] = 1
            # entity_label_data[sub.end-1][sub.start] = 1
            # entity_label_data[obj.end-1][obj.start] = 1

            hh_label_data[sub.start][obj.start] = relation.id
            tt_label_data[sub.end-1][obj.end-1] = relation.id

            entity_relation_value.append((sub.start, sub.end-1, obj.start, obj.end-1, relation.id))

        return {
            "char_encode_id": char_encode_id,
            "word_encode_id": word_encode_id,
            "entity_label_data": entity_label_data,
            "hh_label_data": hh_label_data,
            "tt_label_data": tt_label_data,
            "entity_relation_value": entity_relation_value,
            "text_raw": text_raw,
            "mt_mask": mt_mask,
            "mt_entity_mask": mt_entity_mask
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_encode_id = []
        batch_word_encode_id = []
        batch_entity_label = []
        batch_hh_label_data = []
        batch_tt_label_data = []
        batch_entity_relation_value = []
        batch_text_raw = []
        batch_mt_mask = []
        batch_mt_entity_mask = []

        max_len = 0
        for data in input_batch_data:
            batch_text_raw.append(data["text_raw"])
            batch_char_encode_id.append(data["char_encode_id"])
            batch_word_encode_id.append(data["word_encode_id"])
            batch_entity_relation_value.append(data["entity_relation_value"])

            max_len = max(len(data["char_encode_id"]), max_len)

        for data in input_batch_data:
            entity_label_data = data["entity_label_data"]
            # print(max_len, entity_label_data.shape)
            entity_label_data = np.pad(entity_label_data,
                              ((0,  max_len - entity_label_data.shape[0]), (0, max_len - entity_label_data.shape[1])),
                              'constant', constant_values=0)

            batch_entity_label.append(entity_label_data)
            hh_label_data = data["hh_label_data"]
            hh_label_data = np.pad(hh_label_data,
                                       ((0, max_len - hh_label_data.shape[0]),
                                        (0, max_len - hh_label_data.shape[1])),
                                       'constant', constant_values=0)
            batch_hh_label_data.append(hh_label_data)

            tt_label_data = data["tt_label_data"]
            tt_label_data = np.pad(tt_label_data,
                                   ((0, max_len - tt_label_data.shape[0]),
                                    (0, max_len - tt_label_data.shape[1])),
                                   'constant', constant_values=0)
            batch_tt_label_data.append(tt_label_data)

            mt_mask = data["mt_mask"]
            mt_mask = np.pad(mt_mask,
                                   ((0, max_len - mt_mask.shape[0]),
                                    (0, max_len - mt_mask.shape[1])),
                                   'constant', constant_values=0)
            batch_mt_mask.append(mt_mask)

            mt_entity_mask = data["mt_entity_mask"]
            mt_entity_mask = np.pad(mt_entity_mask,
                             ((0, max_len - mt_entity_mask.shape[0]),
                              (0, max_len - mt_entity_mask.shape[1])),
                             'constant', constant_values=0)
            batch_mt_entity_mask.append(mt_entity_mask)

        batch_char_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_encode_id, maxlen=max_len, padding="post")
        batch_word_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_word_encode_id, maxlen=max_len, padding="post")
        return {
            "char_encode_id": batch_char_encode_id,
            "word_encode_id": batch_word_encode_id,
            "entity_label_data": tf.cast(batch_entity_label, dtype=tf.int32),
            "hh_label_data": tf.cast(batch_hh_label_data, dtype=tf.int32),
            "tt_label_data": tf.cast(batch_tt_label_data, dtype=tf.int32),
            "entity_relation_value": batch_entity_relation_value,
            "text_raw": batch_text_raw,
            "max_len": tf.cast(max_len, dtype=tf.int32),
            "mt_mask": tf.cast(batch_mt_mask, dtype=tf.int32),
            "mt_entity_mask": tf.cast(batch_mt_entity_mask, dtype=tf.int32)
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

def evaluation(batch_data, model, t_batch_num):
    hit_num = 0.0
    true_num = 0.0
    predict_num = 0.0
    entity_logits, hh_logits, tt_logits = model(batch_data["char_encode_id"],
                                                batch_data["word_encode_id"],
                                                None,
                                                batch_data["max_len"])
    print(tf.reduce_max(entity_logits))
    mt_mask = batch_data["mt_mask"].numpy()
    entity_argmax = tf.cast(tf.math.greater_equal(entity_logits, 0.5), dtype=tf.int32)
    b, l, l, v = entity_argmax.shape
    entity_argmax = tf.reshape(entity_argmax, [t_batch_num, l, l]).numpy() * mt_mask
    hh_argmax = tf.argmax(hh_logits, axis=-1).numpy()
    hh_argmax = hh_argmax * mt_mask
    tt_argmax = tf.argmax(tt_logits, axis=-1).numpy()
    tt_argmax = tt_argmax * mt_mask

    for i in range(t_batch_num):
        true_predict = set(batch_data["entity_relation_value"][i])
        true_num += len(true_predict)

        entity_label_multi = entity_argmax[i]
        entity_list = set()
        for iv, srow in enumerate(entity_label_multi):
            for jv, ei in enumerate(srow):
                if jv < iv:
                    continue
                if ei == 1:
                    entity_list.add((iv, jv))
        print("entity", len(entity_list))
        hh_dict = dict()
        hh_rel_multi = hh_argmax[i]
        print(tf.reduce_max(hh_rel_multi))
        for iv, hrow in enumerate(hh_rel_multi):
            for jv, ei in enumerate(hrow):
                if ei == 0:
                    continue
                hh_dict.setdefault(ei, [])
                hh_dict[ei].append((iv, jv))

        print(len(hh_dict))
        tt_dict = dict()
        tt_rel_multi = tt_argmax[i]
        for iv, trow in enumerate(tt_rel_multi):
            for jv, ei in enumerate(trow):
                if ei == 0:
                    continue
                tt_dict.setdefault(ei, [])
                tt_dict[ei].append((iv, jv))
        print(len(tt_dict))

        predict_extract = []
        for kr, h_list in hh_dict.items():
            if kr not in tt_dict:
                continue
            tr_list = tt_dict[kr]
            for hs, ho in h_list:
                for ts, to in tr_list:
                    if (hs, ts) not in entity_list:
                        continue
                    if (ho, to) not in entity_list:
                        continue
                    p_value = (hs, ts, ho, to, kr)
                    predict_extract.append(p_value)
                    predict_num += 1
                    if p_value in true_predict:
                        hit_num += 1

    return {
        "hit_num": hit_num,
        "real_count": true_num,
        "predict_count":  predict_num
    }


def main():
    model = Tplink(char_size, char_embed, word_size, word_embed, lstm_size, rel_num)

    # loss_func = tf.keras.losses.BinaryCrossentropy()

    def loss_func(input_true, input_logits, sample_weight=None):
        cross_func = tf.keras.losses.BinaryCrossentropy()
        mask = tf.math.logical_not(tf.math.equal(input_true, 0))
        mask = tf.where(mask, 5.0, 1.0)
        mask *= tf.cast(sample_weight, dtype=tf.float32)
        # mask = tf.cast(mask, dtype=tf.int64)
        input_true = tf.expand_dims(input_true, axis=-1)
        input_logits = tf.expand_dims(input_logits, axis=-1)
        lossv = cross_func(input_true, input_logits, sample_weight=mask)

        return lossv

    def loss_funv(input_true, input_logits, sample_weight=None):
        cross_func = tf.keras.losses.SparseCategoricalCrossentropy()
        mask = tf.math.logical_not(tf.math.equal(input_true, 0))
        mask = tf.where(mask, 10.0, 1.0)
        mask *= tf.cast(sample_weight, dtype=tf.float32)
        # mask = tf.cast(mask, dtype=tf.int64)
        lossv = cross_func(input_true, input_logits, sample_weight=mask)

        return lossv

    optimizer = tf.keras.optimizers.Adam()

    @tf.function(experimental_relax_shapes=True)
    def train_step(input_char_id, input_word_id, input_entity_label, input_hh_label, input_tt_label, input_max_len, input_mt_mask, input_emt_mask):
        with tf.GradientTape() as tape:
            entity_logits, hh_logits, tt_logits = model(input_char_id, input_word_id, input_entity_label, input_max_len, training=True)
            # input_entity_label = tf.expand_dims(input_entity_label, -1)
            # entity_logits = tf.expand_dims(entity_logits, -1)
            loss_v1 = loss_func(input_entity_label, entity_logits, sample_weight=input_emt_mask)
            loss_v2 = loss_funv(input_hh_label, hh_logits, sample_weight=input_mt_mask)
            loss_v3 = loss_funv(input_tt_label, tt_logits, sample_weight=input_mt_mask)
            loss_v = 2*loss_v1 + loss_v2 + loss_v3

            variables = model.variables
            gradients = tape.gradient(loss_v, variables)
            for grad, var in list(zip(gradients, variables)):
                tf.summary.histogram(var.name + '/gradient', grad)
            optimizer.apply_gradients(zip(gradients, variables))
        return loss_v

    data_iter = DataIter(data_loader)
    epoch = 20
    for ep in range(epoch):
        for batch_i, batch_data in enumerate(data_iter.train_iter(batch_num)):
            loss_value = train_step(batch_data["char_encode_id"],
                                    batch_data["word_encode_id"],
                                    batch_data["entity_label_data"],
                                    batch_data["hh_label_data"],
                                    batch_data["tt_label_data"],
                                    batch_data["max_len"],
                                    batch_data["mt_mask"],
                                    batch_data["mt_entity_mask"])
            if batch_i % 100 == 0:
                print("epoch {0} batch {1} loss value {2}".format(ep, batch_i, loss_value))
                print(evaluation(batch_data, model, batch_num))

    eval_res = {
        'hit_num': 0.0, 'real_count': 0.0, 'predict_count': 0.0
    }
    for batch_i, batch_data in enumerate(data_iter.dev_iter(batch_num)):
        e_res = evaluation(batch_data, model, batch_num)
        eval_res["hit_num"] += e_res["hit_num"]
        eval_res["real_count"] += e_res["real_count"]
        eval_res["predict_count"] += e_res["predict_count"]

    print(eval_metrix(eval_res["hit_num"], eval_res["real_count"], eval_res["predict_count"]))



if __name__ == "__main__":
    main()
