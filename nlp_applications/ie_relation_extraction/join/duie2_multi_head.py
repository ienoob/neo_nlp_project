#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/29 21:39
    @Author  : jack.li
    @Site    : 
    @File    : duie2_multi_head.py

"""
import jieba
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from nlp_applications.ner.evaluation import extract_entity
from nlp_applications.ie_relation_extraction.join.multi_head import MultiHeaderModel

data_path = "D:\data\百度比赛\\2021语言与智能技术竞赛：多形态信息抽取任务\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)
triple_regularity = data_loader.triple_set
entity_bio_encoder = {"O": 0}

for i in range(1, len(data_loader.entity2id)):
    entity_bio_encoder["B-{}".format(i)] = len(entity_bio_encoder)
    entity_bio_encoder["I-{}".format(i)] = len(entity_bio_encoder)
entity_bio_id2encoder = {v:k for k, v in entity_bio_encoder.items()}

batch_num = 5
char_size = len(data_loader.char2id)
word_size = len(data_loader.word2id)
char_embed = 32
word_embed = 32
entity_num = len(entity_bio_encoder)
entity_embed_size = 32
rel_num = len(data_loader.relation2id)
rel_embed_size = 32


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
        entity_label_data = np.zeros(len(text_raw))
        rel_label_data = np.zeros((len(text_raw), len(text_raw)))

        entity_relation_value = []
        for relation in doc.relation_list:
            sub = relation.sub
            obj = relation.obj

            e_label = "B-{}".format(sub.id)
            entity_label_data[sub.start] = entity_bio_encoder[e_label]
            for iv in range(sub.start+1, sub.end):
                e_label = "I-{}".format(sub.id)
                entity_label_data[iv] = entity_bio_encoder[e_label]

            e_label = "B-{}".format(obj.id)
            entity_label_data[obj.start] = entity_bio_encoder[e_label]
            for iv in range(obj.start + 1, obj.end):
                e_label = "I-{}".format(obj.id)
                entity_label_data[iv] = entity_bio_encoder[e_label]

            rel_label_data[sub.end-1][obj.end-1] = relation.id
            entity_relation_value.append((sub.id, sub.start, sub.end-1, obj.id, obj.start, obj.end-1, relation.id))

        return {
            "char_encode_id": char_encode_id,
            "word_encode_id": word_encode_id,
            "entity_label_data": entity_label_data,
            "rel_label_data": rel_label_data,
            "entity_relation_value": entity_relation_value,
            "text_raw": text_raw
        }

    def padding_batch_data(self, input_batch_data):
        batch_char_encode_id = []
        batch_word_encode_id = []
        batch_entity_label = []
        batch_rel_label = []
        batch_entity_relation_value = []
        batch_text_raw = []

        max_len = 0
        for data in input_batch_data:
            batch_text_raw.append(data["text_raw"])
            batch_char_encode_id.append(data["char_encode_id"])
            batch_word_encode_id.append(data["word_encode_id"])
            batch_entity_label.append(data["entity_label_data"])
            batch_entity_relation_value.append(data["entity_relation_value"])

            max_len = max(len(data["char_encode_id"]), max_len)

        for data in input_batch_data:
            rel_label_data = data["rel_label_data"]
            rel_label_data = np.pad(rel_label_data,
                              ((0, max_len - rel_label_data.shape[0]), (0, max_len - rel_label_data.shape[1])),
                              'constant', constant_values=0)
            batch_rel_label.append(rel_label_data)
        batch_char_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_char_encode_id, maxlen=max_len, padding="post")
        batch_word_encode_id = tf.keras.preprocessing.sequence.pad_sequences(batch_word_encode_id, maxlen=max_len, padding="post")
        batch_entity_label = tf.keras.preprocessing.sequence.pad_sequences(batch_entity_label, maxlen=max_len, padding="post")
        return {
            "char_encode_id": batch_char_encode_id,
            "word_encode_id": batch_word_encode_id,
            "entity_label_data": batch_entity_label,
            "rel_label_data": tf.cast(batch_rel_label, dtype=tf.int32),
            "max_len": tf.cast(max_len, dtype=tf.int32),
            "entity_relation_value": batch_entity_relation_value,
            "text_raw": batch_text_raw
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


def assert_eval(input_entity_id, input_rel_id, input_entity_relation_value, input_text_raw):
    i_batch_num = input_entity_id.shape[0]
    for ib in range(i_batch_num):
        print(input_text_raw[ib])
        entity_id = [entity_bio_id2encoder[e] for e in input_entity_id[ib]]
        entity_e_list = extract_entity(entity_id)
        print(entity_e_list)
        entity_map = {e_value[1] - 1: e_value for e_value in entity_e_list}
        print(entity_map)

        rel_metric = input_rel_id[ib].numpy()
        rel_list = []
        for iv, o_rel_id_row in enumerate(rel_metric):
            if iv not in entity_map:
                continue
            sub_iv = entity_map[iv]
            for jv, o_rel in enumerate(o_rel_id_row):
                if o_rel == 0:
                    continue
                if iv == jv:
                    continue
                if jv not in entity_map:
                    continue
                obj_jv = entity_map[jv]
                one = (int(sub_iv[2]), sub_iv[0], sub_iv[1]-1,  int(obj_jv[2]), obj_jv[0], obj_jv[1]-1, o_rel)
                rel_list.append(one)
        print(rel_list)

        print(input_entity_relation_value[ib])

def evaluation(input_char_id, input_word_id, input_entity_relation_value, input_model):
    o_entity_logits, o_rel_logits, _ = input_model(input_char_id, input_word_id)
    o_entity_id = tf.argmax(o_entity_logits, axis=-1)

    o_rel_ids = tf.argmax(o_rel_logits, axis=-1)
    i_batch_num = o_entity_logits.shape[0]
    hit_num = 0.0
    real_count = 0.0
    predict_count = 0.0
    for ib in range(i_batch_num):
        entity_id = [entity_bio_id2encoder[e] for e in o_entity_id[ib].numpy()]
        entity_e_list = extract_entity(entity_id)
        print("entity_num {}".format(len(entity_e_list)))

        entity_map = {e_value[1]-1: e_value for e_value in entity_e_list}
        # print("entity", entity_e_list)
        # entity_e_list = [ for s, e, si in entity_e_list]
        o_rel_id = o_rel_ids[ib].numpy()
        rel_list = []
        real_count += len(input_entity_relation_value[ib])

        real_data_set = set(input_entity_relation_value[ib])
        for iv, o_rel_id_row in enumerate(o_rel_id):
            if iv not in entity_map:
                continue
            sub_iv = entity_map[iv]
            for jv, o_rel in enumerate(o_rel_id_row):
                if o_rel == 0:
                    continue
                if iv == jv:
                    continue
                if jv not in entity_map:
                    continue
                obj_jv = entity_map[jv]
                if (int(sub_iv[2]), o_rel, int(obj_jv[2])) not in triple_regularity:
                    continue
                one = (int(sub_iv[2]), sub_iv[0], sub_iv[1]-1,  int(obj_jv[2]), obj_jv[0], obj_jv[1]-1, o_rel)
                rel_list.append(one)
                predict_count += 1
                if one in real_data_set:
                    hit_num += 1
        print("relation", rel_list)
        print("real", input_entity_relation_value[ib])
    res = {
        "hit_num": hit_num,
        "real_count": real_count,
        "predict_count":  predict_count
    }

    return res


def main():
    # test_run_model()
    model = MultiHeaderModel(char_size, char_embed, word_size, entity_num, entity_embed_size, rel_num, rel_embed_size)

    def loss_func1(input_y, logits):
        cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(input_y, 0))

        mask = tf.cast(mask, dtype=tf.int64)
        lossv = cross_func(input_y, logits, sample_weight=mask)

        return lossv
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function(experimental_relax_shapes=True)
    def train_step(input_char_id, input_word_id, input_entity_id, input_rel_id, data_max_len):
        with tf.GradientTape() as tape:
            o_entity_logits, o_rel_logits, entity_label_mask = model(input_char_id, input_word_id, input_entity_id, data_max_len,
                                                  training=True)
            loss_v = loss_func(input_entity_id, o_entity_logits, sample_weight=entity_label_mask) + loss_func1(input_rel_id, o_rel_logits)

            variables = model.variables
            gradients = tape.gradient(loss_v, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        return loss_v

    data_iter = DataIter(data_loader)
    epoch = 20
    for ep in range(epoch):
        for batch_i, batch_data in enumerate(data_iter.train_iter(batch_num)):

            loss_value = train_step(batch_data["char_encode_id"],
                                    batch_data["word_encode_id"],
                                    batch_data["entity_label_data"],
                                    batch_data["rel_label_data"],
                                    batch_data["max_len"])
            if batch_i % 100 == 0:
                print("epoch {0} batch {1} loss value {2}".format(ep, batch_i, loss_value))
                print(evaluation(batch_data["char_encode_id"],
                                 batch_data["word_encode_id"], batch_data["entity_relation_value"], model))
                # assert_eval(batch_data["entity_label_data"], batch_data["rel_label_data"],
                #               batch_data["entity_relation_value"],
                #               batch_data["text_raw"])

    dev_bt_num = 10
    evaluation_res = {
        "hit_num": 0,
        "real_count": 0,
        "predict_count":  0}
    for batch_data in data_iter.dev_iter(dev_bt_num):
        sub_eval_res = evaluation(batch_data["char_encode_id"],
                   batch_data["word_encode_id"], batch_data["entity_relation_value"], model)
        evaluation_res["hit_num"] += sub_eval_res["hit_num"]
        evaluation_res["real_count"] += sub_eval_res["real_count"]
        evaluation_res["predict_count"] += sub_eval_res["predict_count"]

    evaluation_res["recall"] = (evaluation_res["hit_num"] + 1e-8) / (evaluation_res["real_count"] + 1e-3)
    evaluation_res["precision"] = (evaluation_res["hit_num"] + 1e-8) / (evaluation_res["predict_count"] + 1e-3)

    evaluation_res["f1_value"] = 2 * evaluation_res["recall"] * evaluation_res["precision"] / (evaluation_res["recall"] + evaluation_res["precision"])

    print(evaluation_res)


if __name__ == "__main__":
    main()
