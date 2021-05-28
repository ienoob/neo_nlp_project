#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:42
    @Author  : jack.li
    @Site    : 
    @File    : sp_ert.py

    实现 Span-based Joint Entity and Relation Extraction with Transformer Pre-training


    参考 https://github.com/lavis-nlp/spert
"""
import json
import pickle
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document


def batch_index(first_list, index_list):
    batch_num = first_list.shape[0]
    relation_num = index_list.shape[1]

    new_data = []
    for i in range(batch_num):
        batch_data = []
        for j in range(relation_num):
            batch_data.append(tf.gather(first_list, axis=1, indices=index_list[i][j]))
        new_data.append(batch_data)

    return tf.cast(new_data, dtype=tf.float32)



def build_entity_feature(input_embed, entity_mask, input_size: tf.Tensor):
    inner_batch_num = input_embed.shape[0]
    input_size = tf.reshape(input_size, (inner_batch_num,))
    entity_feature = tf.repeat(input_embed, input_size, axis=0)
    # for i in range(inner_batch_num):
    #     for v in range(input_size[i]):
    #         entity_feature.append(input_embed[i])
    # entity_feature = tf.cast(entity_feature, dtype=tf.float32)
    m = tf.cast(tf.expand_dims(entity_mask, -1) == 0, tf.float32) * (-1e1)
    entity_spans_pool = m + entity_feature
    entity_spans_pool = tf.reduce_max(entity_spans_pool, axis=1)

    return entity_spans_pool


def build_relation_feature(input_embed, entity_spans_pool, input_relation_entity,
                           size_embeddings, rel_mask, input_num, relation_count=0):
    inner_batch_num = input_embed.shape[0]
    input_num = tf.reshape(input_num, (inner_batch_num, ))
    relation_embed_feature = tf.repeat(input_embed, input_num, axis=0)

    def new_func(index_list):
        n_shape = entity_spans_pool.shape[1] * 2
        target_tensor = tf.reshape(tf.gather(entity_spans_pool, index_list), (n_shape,))

        return target_tensor

    def new_func2(index_list):
        n_shape = size_embeddings.shape[1] * 2
        target_tensor = tf.reshape(tf.gather(size_embeddings, index_list), (n_shape,))
        return target_tensor

    relation_entity_feature = tf.map_fn(new_func, input_relation_entity, dtype=tf.float32)
    relation_size_feature = tf.map_fn(new_func2, input_relation_entity, dtype=tf.float32)
    # relation_entity_feature = tf.cast(relation_entity_feature, dtype=tf.float32)
    # relation_size_feature = tf.cast(relation_size_feature, dtype=tf.float32)


    # relation_count = input_relation_entity.shape[0]

    # for iv in range(relation_count):
    #     ind = input_relation_entity[iv]
    #     # relation_embed_feature.append(input_embed[i])
    #     relation_entity_pair = new_func(entity_spans_pool, ind)
    #     relation_entity_feature.append(relation_entity_pair)
    #     relation_size_pair = new_func(size_embeddings, ind)
    #     relation_size_feature.append(relation_size_pair)

    m = tf.cast(tf.expand_dims(rel_mask, -1) == 0, tf.float32) * (-1e1)
    relation_spans_pool = m + relation_embed_feature
    relation_embed = tf.reduce_max(relation_spans_pool, axis=1)
    # relation_entity_featurev = tf.stack(relation_entity_feature)
    # relation_size_featurev = tf.stack(relation_size_feature)
    relation_feature = tf.concat([relation_embed, relation_entity_feature, relation_size_feature], axis=1)

    return relation_feature



class SpERt(tf.keras.models.Model):

    def __init__(self, args, relation_types, entity_types, max_pairs):
        super(SpERt, self).__init__()

        self.embed = tf.keras.layers.Embedding(args.char_size, args.embed_size)
        self.size_embed = tf.keras.layers.Embedding(args.size_value, args.size_embed_size)
        # 相比于原始模型，增加双向lstm 层
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.lstm_size, return_sequences=True))
        self.rel_classifier = tf.keras.layers.Dense(args.relation_type, activation="softmax")
        self.entity_classifier = tf.keras.layers.Dense(args.entity_type, activation="softmax")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.rel_dropout = tf.keras.layers.Dropout(0.5)

        # self.cla
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

    def call(self, text_ids, text_contexts, entity_masks, entity_sizes, entity_nums, relations_entity, rel_masks,  relation_nums,
             training=None, mask=None):
        # z这里用普通embed替代bert, 这是为了正常运行，毕竟╮(╯▽╰)╭内存太小了
        h = self.embed(text_ids)
        h = self.bi_lstm(h)

        size_embeddings = self.size_embed(entity_sizes)
        entity_spans_pool = build_entity_feature(h, entity_masks, entity_nums)
        entity_repr = tf.concat([entity_spans_pool, size_embeddings], axis=1)
        entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)
        relation_count = relations_entity.shape[0]
        relation_feature = build_relation_feature(h, entity_spans_pool, relations_entity, size_embeddings, rel_masks, relation_nums, relation_count)
        rel_clf = self.rel_classifier(relation_feature)

        return entity_clf, rel_clf

    def filter_span(self, input_entity_logits, input_entity_start_end, input_max_len):
        entity_list = tf.argmax(input_entity_logits, axis=-1)
        entity_list = entity_list.numpy()

        entity_span_list = []
        for i, ix in enumerate(entity_list):
            # entity_span_list.append(i)
            if ix != tf.cast(0, dtype=tf.int64):
                entity_span_list.append(i)
        # entity_span_list = entity_span_list[:10]
        relations_entity = []
        relation_masks = []
        relations_num = 0
        for i in entity_span_list:
            ss, se = input_entity_start_end[i]
            for j in entity_span_list:
                os, oe = input_entity_start_end[j]
                if i == j:
                    continue
                if (entity_list[i], entity_list[j]) not in data_loader.entity_couple_set:
                    continue
                start = se if se > os else oe
                end = os if se > os else ss
                if start > end:
                    start, end = end, start

                relation_maskv = [1 if ind >= start and ind < end else 0 for ind in range(input_max_len)]
                relation_masks.append(relation_maskv)
                relations_entity.append([i, j])
                relations_num += 1

        return tf.cast(relations_entity, dtype=tf.int64), tf.cast(relation_masks, dtype=tf.int64), tf.cast([[relations_num]], tf.int64)

    def predict(self, text_ids, text_contexts, entity_masks, entity_sizes, entity_nums, input_entity_start_end, input_max_len):
        h = self.embed(text_ids)
        size_embeddings = self.size_embed(entity_sizes)
        entity_spans_pool = build_entity_feature(h, entity_masks, entity_nums)
        entity_repr = tf.concat([entity_spans_pool, size_embeddings], axis=1)
        entity_repr = self.dropout(entity_repr)
        entity_clf = self.entity_classifier(entity_repr)

        entity_list = tf.argmax(entity_clf, axis=-1)
        entity_list = entity_list.numpy()

        entity_nums_numpy = entity_nums.numpy()
        start = 0
        batch_res = []
        for e_num in entity_nums_numpy:

            rel_res = []
            relations_entity, relation_mask, relation_nums = self.filter_span(entity_clf[start:e_num[0]], input_entity_start_end[start:e_num[0]], input_max_len)
            if relations_entity.shape[0]:
                relation_feature = build_relation_feature(h, entity_spans_pool, relations_entity, size_embeddings, relation_mask,
                                                          relation_nums)
                rel_clf = self.rel_classifier(relation_feature)
                rel_clf_argmax = tf.argmax(rel_clf, axis=-1)

                rel_res = [(relations_entity[i].numpy()[0], relations_entity[i].numpy()[1], label) for i, label in enumerate(rel_clf_argmax.numpy())]
                rel_res = [(input_entity_start_end[si], entity_list[si], input_entity_start_end[oi], entity_list[oi], p) for si, oi, p in rel_res if p]

            batch_res.append(rel_res)
            start = e_num[0]
        return batch_res



spert = SpERt(relation_type, entity_type, 10)

# sample_entity_res, sample_relation_res = spert(sample_encoding, sample_mask, sample_entity_mask, sample_entity_sizes, sample_relation, sample_relation_mask)


# print(sample_entity_res.shape)
# print(sample_relation_res.shape)

sample_entity_label = tf.constant([[1, 2, 3]])
sample_relation_label = tf.constant(([[1]]))

# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
loss_f = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_f2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# print(loss(sample_entity_label, sample_entity_res))
# print(loss(sample_relation_label, sample_relation_res))


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

epoch = 10
for e in range(epoch):
    batch_data_iter = get_sample_data(batch_num)
    for i, batch_data in enumerate(batch_data_iter):
        lossv = train_step(batch_data["encodings"],
                           batch_data["context_masks"],
                           batch_data["entity_masks"],
                           batch_data["entity_sizes"],
                           batch_data["entity_num"],
                           batch_data["relation_entity"],
                           batch_data["rel_masks"],
                           batch_data["relation_num"],
                           batch_data["entity_spans"],
                           batch_data["relations"])

        if i % 100 == 0:
            print("epoch {0} batch {1} loss value is {2}".format(e, i, lossv))
            spert.save_weights(model_path, save_format='tf')


test_batch_num = 1
spert.load_weights(model_path)
batch_data_iter = get_test_sample_data(test_batch_num)
submit_res = []
batch_i = 0
save_path = "D:\\tmp\submit_data\\duie.json"


for i, batch_data in enumerate(batch_data_iter):
    print("batch {} start".format(i))
    pres = spert.predict(batch_data["encodings"],
                               batch_data["context_masks"],
                               batch_data["entity_masks"],
                               batch_data["entity_sizes"],
                               batch_data["entity_num"],
                               batch_data["entity_start_end"],
                               batch_data["max_len"])
    for j in range(test_batch_num):
        n_pres = pres[j]
        doc = data_loader.test_documents[i*test_batch_num+j]
        i_text = doc.raw_text
        spo_list = []
        for sop in n_pres:
            sub_i, sub_j = sop[0]
            sub_type = sop[1]
            obj_i, obj_j = sop[2]
            obj_type = sop[3]
            pre_type = sop[4]

            if (sub_type, pre_type, obj_type) not in data_loader.triple_set:
                continue
            spo_list.append({
                "predicate": data_loader.id2relation[pre_type],
                "subject": i_text[sub_i:sub_j],
                "subject_type": data_loader.id2entity[sub_type],
                "object": {
                    "@value": i_text[obj_i:obj_j],
                },
                "object_type": {
                    "@value": data_loader.id2entity[obj_type],
                }
            })

        single_spo = {
            "text": i_text,
            "spo_list": spo_list
        }
        print(single_spo)
    #     submit_res.append(json.dumps(single_spo))
    #
    # with open(save_path, "a+") as f:
    #     f.write("\n".join(submit_res))
    submit_res = []

# save_batch_num = 1000

# with open(save_path, "w") as f:
#     f.write("\n".join(submit_res[:save_batch_num]))
#
# batch_count = int(len(submit_res)//save_batch_num) + 1
# print(batch_count)
#
# for ib in range(1, batch_count):
#     ib_start = ib*save_batch_num
#     ib_end = ib*save_batch_num + save_batch_num
#     if len(submit_res[ib_start:ib_end]) == 0:
#         break
#     with open(save_path, "a+") as f:
#         f.write("\n")
#         f.write("\n".join(submit_res[ib_start:ib_end]))




