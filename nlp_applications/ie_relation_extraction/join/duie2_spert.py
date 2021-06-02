#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document
from nlp_applications.ie_relation_extraction.join.spert import SpERt
from nlp_applications.data_loader import BaseDataIterator


batch_num = 2
entity_max_len = 70
random_choice_num = 100
random_relation_num = 20

data_path = "D:\\data\\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)

entity_couple_set = data_loader.entity_couple_set


class DataIter(BaseDataIterator):

    def __init__(self, input_loader):
        super(DataIter, self).__init__(input_loader)

    def single_doc_processor(self, doc: Document):
        text_ids = doc.text_id
        text_len = len(text_ids)

        entity_list = doc.entity_list
        relation_list = doc.relation_list

        entity_span = []
        entity_span_list = []
        sub_entity_masks = []
        sub_entity_size = []
        entity_loc_set = set()
        for entity in entity_list:
            entity_mask = [1 if ind >= entity.start and ind < entity.end else 0 for ind in range(text_len)]
            sub_entity_masks.append(entity_mask)
            entity_span.append(entity.id)
            entity_span_list.append(entity)
            sub_entity_size.append(entity.size)
            entity_loc_set.add((entity.start, entity.end))

        negative_entity_data = []
        for j in range(text_len - 1):
            for k in range(j + 1, text_len):
                if (j, k) in entity_loc_set:
                    continue
                if k - j > entity_max_len:
                    break
                entity_mask = [1 if ind >= j and ind < k else 0 for ind in range(text_len)]
                negative_entity_data.append((entity_mask, 0, k - j))
                # sub_entity_masks.append(entity_mask)
                # entity_span.append(0)
                # sub_entity_size.append(k - j)

        entity_num = len(negative_entity_data)
        if entity_num > random_choice_num:
            random_inx = np.random.choice(entity_num, random_choice_num)
            negative_entity_data = [negative_entity_data[ind] for ind in random_inx]
        for entity_mask, e_id, e_size in negative_entity_data:
            sub_entity_masks.append(entity_mask)
            entity_span.append(e_id)
            sub_entity_size.append(e_size)

        entity_d = dict()

        relation_entity_data = dict()
        relation_entity_set = set()
        entity_relation_value = []
        for rl in relation_list:
            relation_entity_data[(rl.sub, rl.obj)] = rl.id
            relation_entity_set.add(rl.sub)
            entity_d[rl.sub.id] = rl.sub
            relation_entity_set.add(rl.obj)
            entity_d[rl.obj.id] = rl.obj

            sub = rl.sub
            obj = rl.obj
            entity_relation_value.append((sub.id, sub.start, sub.end - 1, obj.id, obj.start, obj.end - 1, rl.id))

        relation_entity_span = []
        relation_entity_list = list(relation_entity_set)
        relation_label = []
        relation_mask = []
        negative_relation_data = []

        for i, ei in enumerate(relation_entity_list):

            for j, ej in enumerate(relation_entity_list):
                if i == j:
                    continue

                sub = ei
                obj = ej

                start = sub.end if sub.end > obj.start else obj.end
                end = obj.start if sub.end > obj.start else sub.start
                if start > end:
                    start, end = end, start

                relation_maskv = [1 if ind >= start and ind < end else 0 for ind in range(text_len)]
                if (ei, ej) in relation_entity_data:

                    relation_entity_span.append((entity_span_list.index(ei), entity_span_list.index(ej)))
                    relation_label.append(relation_entity_data[(ei, ej)])
                    relation_mask.append(relation_maskv)
                else:
                    negative_relation_data.append(
                        (entity_span_list.index(ei), entity_span_list.index(ej), 0, relation_maskv))

        negative_num = len(negative_relation_data)
        if negative_num > random_relation_num:
            random_inx = np.random.choice(negative_num, random_relation_num)
            negative_relation_data = [negative_relation_data[ind] for ind in random_inx]

        for e1, e2, rl, rm in negative_relation_data:
            relation_entity_span.append((e1, e2))
            relation_label.append(rl)
            relation_mask.append(rm)

        return {
            "encoding": text_ids,
            "context_mask": [1 for _ in text_ids],
            "entity_span": entity_span,
            "entity_mask": sub_entity_masks,
            "entity_size": sub_entity_size,
            "relation_entity_spans": relation_entity_span,
            "relation_labels": relation_label,
            "relation_masks": tf.cast(relation_mask, dtype=tf.int32),
            "entity_relation_value": entity_relation_value
        }

    def sample_single_predict_data(doc: Document):
        text_ids = doc.text_id
        text_len = len(text_ids)

        entity_span = []
        sub_entity_masks = []
        sub_entity_size = []
        sub_entity_start_end = []

        for j in range(text_len - 1):
            for k in range(j + 1, text_len):
                if k - j > entity_max_len:
                    break
                entity_mask = [1 if ind >= j and ind < k else 0 for ind in range(text_len)]
                sub_entity_masks.append(entity_mask)
                entity_span.append(0)
                sub_entity_start_end.append((j, k))
                sub_entity_size.append(k - j)

        return {
            "encoding": text_ids,
            "context_mask": [1 for _ in text_ids],
            "entity_span": entity_span,
            "entity_mask": sub_entity_masks,
            "entity_size": sub_entity_size,
            "entity_start_end": sub_entity_start_end
        }

    def single_dev_data(self, doc: Document):
        text_ids = doc.text_id
        text_len = len(text_ids)
        relation_list = doc.relation_list

        entity_span = []
        sub_entity_masks = []
        sub_entity_size = []
        sub_entity_start_end = []

        entity_relation_value = []
        for rl in relation_list:
            sub = rl.sub
            obj = rl.obj
            entity_relation_value.append((sub.id, sub.start, sub.end, obj.id, obj.start, obj.end, rl.id))

        for j in range(text_len - 1):
            for k in range(j + 1, text_len):
                if k - j > entity_max_len:
                    break
                entity_mask = [1 if ind >= j and ind < k else 0 for ind in range(text_len)]
                sub_entity_masks.append(entity_mask)
                entity_span.append(0)
                sub_entity_start_end.append((j, k))
                sub_entity_size.append(k - j)

        return {
            "encoding": text_ids,
            "context_mask": [1 for _ in text_ids],
            "entity_span": entity_span,
            "entity_mask": sub_entity_masks,
            "entity_size": sub_entity_size,
            "entity_start_end": sub_entity_start_end,
            "entity_relation_value": entity_relation_value
        }

    def padding_batch_data(self, input_batch_data):
        batch_encodings = []
        batch_context_mask = []
        batch_entity_span = []
        batch_entity_masks = []
        batch_entity_sizes = []
        batch_relations = []
        batch_rel_masks = []
        batch_entity_num = []
        batch_relation_num = []
        batch_relation_entity = []
        batch_entity_relation_value = []

        max_len = 0
        for data in input_batch_data:
            batch_encodings.append(data["encoding"])
            batch_context_mask.append(data["context_mask"])

            batch_entity_num.append(len(data["entity_span"]))
            batch_relation_num.append(len(data["relation_labels"]))
            # batch_relation_entity.append(data["relation_entity_spans"])
            max_len = max(max_len, len(data["encoding"]))
            batch_entity_relation_value.append(data["entity_relation_value"])

        for data in input_batch_data:

            for entity_mask in data["entity_mask"]:
                batch_entity_masks.append(entity_mask)
            for relation in data["relation_labels"]:
                batch_relations.append(relation)
            for rel_mask in data["relation_masks"]:
                batch_rel_masks.append(rel_mask)
            for entity_size in data["entity_size"]:
                batch_entity_sizes.append(entity_size)

            for entity_s in data["entity_span"]:
                batch_entity_span.append(entity_s)

            for relation_pair in data["relation_entity_spans"]:
                batch_relation_entity.append(relation_pair)
        batch_encodings = tf.keras.preprocessing.sequence.pad_sequences(batch_encodings, padding="post")
        batch_encodings = tf.cast(batch_encodings, dtype=tf.int32)
        batch_context_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_context_mask, padding="post")
        batch_context_mask = tf.cast(batch_context_mask, dtype=tf.int32)
        batch_entity_masks = tf.keras.preprocessing.sequence.pad_sequences(batch_entity_masks, padding="post")
        batch_entity_masks = tf.cast(batch_entity_masks, dtype=tf.float32)
        batch_rel_masks = tf.keras.preprocessing.sequence.pad_sequences(batch_rel_masks, padding="post")
        batch_rel_masks = tf.cast(batch_rel_masks, dtype=tf.float32)

        return {
            "encodings": batch_encodings,
            "context_masks": batch_context_mask,
            "entity_spans": tf.reshape(tf.cast(batch_entity_span, dtype=tf.int32), (len(batch_entity_span), 1)),
            "entity_masks": batch_entity_masks,
            "entity_sizes": tf.cast(batch_entity_sizes, dtype=tf.int32),
            "entity_num": tf.reshape(tf.cast(batch_entity_num, dtype=tf.int32), (len(batch_entity_num), 1)),
            "relations": tf.reshape(tf.cast(batch_relations, dtype=tf.int32), (len(batch_relations), 1)),
            "rel_masks": batch_rel_masks,
            "relation_entity": tf.reshape(tf.cast(batch_relation_entity, dtype=tf.int32),
                                          (len(batch_relation_entity), 2)),
            "relation_num": tf.reshape(tf.cast(batch_relation_num, dtype=tf.int32), (len(batch_relation_num), 1)),
            "entity_relation_value": batch_entity_relation_value
        }

    def padding_test_data(self, input_batch_data):
        batch_encodings = []
        batch_context_mask = []
        batch_entity_span = []
        batch_entity_masks = []
        batch_entity_sizes = []
        batch_entity_num = []
        batch_entity_start_end = []

        max_len = 0
        for data in input_batch_data:
            batch_encodings.append(data["encoding"])
            batch_context_mask.append(data["context_mask"])
            batch_entity_num.append(len(data["entity_span"]))

            max_len = max(max_len, len(data["encoding"]))

        for data in input_batch_data:

            for entity_mask in data["entity_mask"]:
                batch_entity_masks.append(entity_mask)
            for entity_size in data["entity_size"]:
                batch_entity_sizes.append(entity_size)

            for entity_s in data["entity_span"]:
                batch_entity_span.append(entity_s)

            for entity_se in data["entity_start_end"]:
                batch_entity_start_end.append(entity_se)

        batch_encodings = tf.keras.preprocessing.sequence.pad_sequences(batch_encodings, padding="post")
        batch_encodings = tf.cast(batch_encodings, dtype=tf.int32)
        batch_context_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_context_mask, padding="post")
        batch_context_mask = tf.cast(batch_context_mask, dtype=tf.int32)
        batch_entity_masks = tf.keras.preprocessing.sequence.pad_sequences(batch_entity_masks, padding="post")
        batch_entity_masks = tf.cast(batch_entity_masks, dtype=tf.int32)

        return {
            "encodings": batch_encodings,
            "context_masks": batch_context_mask,
            "entity_spans": tf.reshape(tf.cast(batch_entity_span, dtype=tf.int32), (len(batch_entity_span), 1)),
            "entity_masks": batch_entity_masks,
            "entity_sizes": tf.cast(batch_entity_sizes, dtype=tf.int32),
            "entity_num": tf.reshape(tf.cast(batch_entity_num, dtype=tf.int32), (len(batch_entity_num), 1)),
            "entity_start_end": batch_entity_start_end,
            "max_len": max_len
        }

    def padding_dev_data(self, input_batch_data):
        batch_encodings = []
        batch_context_mask = []
        batch_entity_span = []
        batch_entity_masks = []
        batch_entity_sizes = []
        batch_entity_num = []
        batch_entity_start_end = []
        batch_entity_relation_value = []

        max_len = 0
        for data in input_batch_data:
            batch_encodings.append(data["encoding"])
            batch_context_mask.append(data["context_mask"])
            batch_entity_num.append(len(data["entity_span"]))

            max_len = max(max_len, len(data["encoding"]))
            batch_entity_relation_value.append(data["entity_relation_value"])

        for data in input_batch_data:

            for entity_mask in data["entity_mask"]:
                batch_entity_masks.append(entity_mask)
            for entity_size in data["entity_size"]:
                batch_entity_sizes.append(entity_size)

            for entity_s in data["entity_span"]:
                batch_entity_span.append(entity_s)

            for entity_se in data["entity_start_end"]:
                batch_entity_start_end.append(entity_se)

        batch_encodings = tf.keras.preprocessing.sequence.pad_sequences(batch_encodings, padding="post")
        batch_encodings = tf.cast(batch_encodings, dtype=tf.int32)
        batch_context_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_context_mask, padding="post")
        batch_context_mask = tf.cast(batch_context_mask, dtype=tf.int32)
        batch_entity_masks = tf.keras.preprocessing.sequence.pad_sequences(batch_entity_masks, padding="post")
        batch_entity_masks = tf.cast(batch_entity_masks, dtype=tf.float32)

        return {
            "encodings": batch_encodings,
            "context_masks": batch_context_mask,
            "entity_spans": tf.reshape(tf.cast(batch_entity_span, dtype=tf.int32), (len(batch_entity_span), 1)),
            "entity_masks": batch_entity_masks,
            "entity_sizes": tf.cast(batch_entity_sizes, dtype=tf.int32),
            "entity_num": tf.reshape(tf.cast(batch_entity_num, dtype=tf.int32), (len(batch_entity_num), 1)),
            "entity_start_end": batch_entity_start_end,
            "max_len": max_len,
            "entity_relation_value": batch_entity_relation_value
        }


    def dev_iter(self, input_batch_num):
        c_batch_data = []
        for doc in self.data_loader.dev_documents:
            c_batch_data.append(self.single_dev_data(doc))
            if len(c_batch_data) == input_batch_num:
                yield self.padding_dev_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_dev_data(c_batch_data)

def evaluation(input_batch_data, input_model):
    pres = input_model.predict(input_batch_data["encodings"],
                               input_batch_data["context_masks"],
                               input_batch_data["entity_masks"],
                               input_batch_data["entity_sizes"],
                               input_batch_data["entity_num"],
                               input_batch_data["entity_start_end"],
                               input_batch_data["max_len"], entity_couple_set)
    # print(pres)
    # print(input_batch_data["entity_relation_value"])
    i_batch_num = len(input_batch_data["encodings"])
    hit_num = 0.0
    real_count = 0.0
    predict_count = 0.0
    for ib in range(i_batch_num):
        print(pres[ib])
        predict_count += len(pres[ib])
        real_count += len(input_batch_data["entity_relation_value"][ib])

        real_data_set = set(input_batch_data["entity_relation_value"][ib])

        for sub_loc, sub_type, obj_loc, obj_type, pre_type in pres[ib]:
            one = (sub_type, sub_loc[0], sub_loc[1], obj_type, obj_loc[0], obj_loc[1], pre_type)
            if one in real_data_set:
                hit_num += 1

    res = {
        "hit_num": hit_num,
        "real_count": real_count,
        "predict_count":  predict_count
    }

    return res

char_size = len(data_loader.char2id)
size_value = 256
embed_size = 64
hidden_size = 64
lstm_size = 64
size_embed_size = 64
relation_type = len(data_loader.relation2id)
entity_type = len(data_loader.entity2id)

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char_size", type=int, default=char_size)
    parser.add_argument("--size_value", type=int, default=size_value)
    parser.add_argument("--embed_size", type=int, default=embed_size)
    parser.add_argument("--hidden_size", type=int, default=hidden_size)
    parser.add_argument("--lstm_size", type=int, default=lstm_size)
    parser.add_argument("--size_embed_size", type=int, default=size_embed_size)
    parser.add_argument("--relation_num", type=int, default=relation_type)
    parser.add_argument("--entity_num", type=int, default=entity_type)
    parser.add_argument("--max_pairs", type=int, default=10)
    args = parser.parse_args()

    spert = SpERt(args)

    optimizer = tf.keras.optimizers.Adam()
    loss_f = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_f2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(experimental_relax_shapes=True)
    def train_step(encodings, context_masks, entity_masks, entity_sizes, entity_num, relation_entity, rel_masks, relation_num, entity_spans, relations):

        with tf.GradientTape() as tape:
            clf_logits, relation_logits = spert(encodings,
                              context_masks,
                              entity_masks,
                              entity_sizes,
                              entity_num,
                              relation_entity,
                              rel_masks,
                              relation_num)
            loss_v1 = loss_f(entity_spans, clf_logits)
            loss_v2 = loss_f2(relations, relation_logits)
            loss_v = loss_v1 + loss_v2

        variables = spert.variables
        gradients = tape.gradient(loss_v, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return loss_v

    model_path = "D:\\tmp\spert_model\\check"
    if os.path.exists(model_path):
        spert.load_weights(model_path)
    data_iter = DataIter(data_loader)
    # epoch = 2
    # for e in range(epoch):
    #     batch_data_iter = data_iter.train_iter(batch_num)
    #     for i, batch_data in enumerate(batch_data_iter):
    #         lossv = train_step(batch_data["encodings"],
    #                            batch_data["context_masks"],
    #                            batch_data["entity_masks"],
    #                            batch_data["entity_sizes"],
    #                            batch_data["entity_num"],
    #                            batch_data["relation_entity"],
    #                            batch_data["rel_masks"],
    #                            batch_data["relation_num"],
    #                            batch_data["entity_spans"],
    #                            batch_data["relations"])
    #
    #         if i % 100 == 0:
    #             # evaluation(batch_data, spert)
    #             print("epoch {0} batch {1} loss value is {2}".format(e, i, lossv))
    #             spert.save_weights(model_path, save_format='tf')


    test_batch_num = 1
    spert.load_weights(model_path)
    evaluation_res = {
        "hit_num": 0,
        "real_count": 0,
        "predict_count": 0}
    batch_data_iter = data_iter.dev_iter(test_batch_num)
    for i, batch_data in enumerate(batch_data_iter):
        print("batch {} start".format(i))
        sub_eval_res = evaluation(batch_data, spert)
        evaluation_res["hit_num"] += sub_eval_res["hit_num"]
        evaluation_res["real_count"] += sub_eval_res["real_count"]
        evaluation_res["predict_count"] += sub_eval_res["predict_count"]

        print(evaluation_res)

    print(evaluation_res)
    # submit_res = []
    # batch_i = 0
    # save_path = "D:\\tmp\submit_data\\duie.json"
    #
    #
    # for i, batch_data in enumerate(batch_data_iter):
    #     print("batch {} start".format(i))
    #     pres = spert.predict(batch_data["encodings"],
    #                                batch_data["context_masks"],
    #                                batch_data["entity_masks"],
    #                                batch_data["entity_sizes"],
    #                                batch_data["entity_num"],
    #                                batch_data["entity_start_end"],
    #                                batch_data["max_len"])
    #     for j in range(test_batch_num):
    #         n_pres = pres[j]
    #         doc = data_loader.test_documents[i*test_batch_num+j]
    #         i_text = doc.raw_text
    #         spo_list = []
    #         for sop in n_pres:
    #             sub_i, sub_j = sop[0]
    #             sub_type = sop[1]
    #             obj_i, obj_j = sop[2]
    #             obj_type = sop[3]
    #             pre_type = sop[4]
    #
    #             if (sub_type, pre_type, obj_type) not in data_loader.triple_set:
    #                 continue
    #             spo_list.append({
    #                 "predicate": data_loader.id2relation[pre_type],
    #                 "subject": i_text[sub_i:sub_j],
    #                 "subject_type": data_loader.id2entity[sub_type],
    #                 "object": {
    #                     "@value": i_text[obj_i:obj_j],
    #                 },
    #                 "object_type": {
    #                     "@value": data_loader.id2entity[obj_type],
    #                 }
    #             })
    #
    #         single_spo = {
    #             "text": i_text,
    #             "spo_list": spo_list
    #         }
    #         print(single_spo)
        #     submit_res.append(json.dumps(single_spo))
        #
        # with open(save_path, "a+") as f:
        #     f.write("\n".join(submit_res))
"""
    log: 
    0.3721, 0.0322, 0.0592  2021-6-2
"""
from nlp_applications.ner.evaluation import eval_metrix

if __name__ == "__main__":
    # main()
    print(eval_metrix(14076, 37825, 437370))
