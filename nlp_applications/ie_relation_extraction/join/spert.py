#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:42
    @Author  : jack.li
    @Site    : 
    @File    : sp_ert.py

    实现 Span-based Joint Entity and Relation Extraction with Transformer Pre-training


"""
import pickle
import numpy as np
import tensorflow as tf
from nlp_applications.data_loader import LoaderDuie2Dataset, Document

batch_num = 2
entity_max_len = 70
random_choice_num = 100
random_relation_num = 20

data_path = "D:\\data\\关系抽取\\"
data_loader = LoaderDuie2Dataset(data_path)

print(len(data_loader.documents))
print(data_loader.entity_max_len)


def convict_data(input_batch_data):
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

    max_len = 0
    for data in input_batch_data:
        batch_encodings.append(data["encoding"])
        batch_context_mask.append(data["context_mask"])

        batch_entity_num.append(len(data["entity_span"]))
        batch_relation_num.append(len(data["relation_labels"]))
        # batch_relation_entity.append(data["relation_entity_spans"])
        max_len = max(max_len, len(data["encoding"]))


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
    batch_entity_masks = tf.cast(batch_entity_masks, dtype=tf.int32)
    batch_rel_masks = tf.keras.preprocessing.sequence.pad_sequences(batch_rel_masks, padding="post")
    batch_rel_masks = tf.cast(batch_rel_masks, dtype=tf.int32)


    return {
        "encodings": batch_encodings,
        "context_masks": batch_context_mask,
        "entity_spans": tf.reshape(tf.cast(batch_entity_span, dtype=tf.int32), (len(batch_entity_span), 1)),
        "entity_masks": batch_entity_masks,
        "entity_sizes": tf.cast(batch_entity_sizes, dtype=tf.int32),
        "entity_num": tf.reshape(tf.cast(batch_entity_num, dtype=tf.int32), (len(batch_entity_num), 1)),
        "relations": tf.reshape(tf.cast(batch_relations, dtype=tf.int32), (len(batch_relations), 1)),
        "rel_masks": batch_rel_masks,
        "relation_entity": tf.reshape(tf.cast(batch_relation_entity, dtype=tf.int32), (len(batch_relation_entity), 2)),
        "relation_num": tf.reshape(tf.cast(batch_relation_num, dtype=tf.int32), (len(batch_relation_num), 1))
    }

def convict_predict_data(input_batch_data):
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


def sample_single_data(doc: Document):
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
            if k-j > entity_max_len:
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
    for rl in relation_list:
        relation_entity_data[(rl.sub, rl.obj)] = rl.id
        relation_entity_set.add(rl.sub)
        entity_d[rl.sub.id] = rl.sub
        relation_entity_set.add(rl.obj)
        entity_d[rl.obj.id] = rl.obj

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
                negative_relation_data.append((entity_span_list.index(ei), entity_span_list.index(ej), 0, relation_maskv))

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


def get_sample_data(input_batch_num):

    inner_batch_data = []
    for doc in data_loader.documents:
        inner_batch_data.append(sample_single_data(doc))
        if len(inner_batch_data) == input_batch_num:
            yield convict_data(inner_batch_data)
            inner_batch_data = []
    if inner_batch_data:
        yield convict_data(inner_batch_data)


def get_test_sample_data(input_batch_num):

    inner_batch_data = []
    for doc in data_loader.test_documents:
        inner_batch_data.append(sample_single_predict_data(doc))
        if len(inner_batch_data) == input_batch_num:
            yield convict_predict_data(inner_batch_data)
            inner_batch_data = []
    if inner_batch_data:
        yield convict_predict_data(inner_batch_data)


char_size = len(data_loader.char2id)
size_value = 256
embed_size = 64
hidden_size = 64
size_embed_size = 64
relation_type = len(data_loader.relation2id)
entity_type = len(data_loader.entity2id)


def get_token(h, x, token):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    # token_h = h.view(-1, emb_size)
    token_h = tf.reshape(h, (-1, emb_size))
    # tf.conti
    # flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    # token_h = token_h[flat == token, :]

    return token_h


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

    def __init__(self, relation_types, entity_types, max_pairs):
        super(SpERt, self).__init__()

        self.embed = tf.keras.layers.Embedding(char_size, embed_size)
        self.size_embed = tf.keras.layers.Embedding(size_value, size_embed_size)
        self.rel_classifier = tf.keras.layers.Dense(relation_type, activation="softmax")
        self.entity_classifier = tf.keras.layers.Dense(entity_type, activation="softmax")
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

        entity_span_list = []
        for i, ix in enumerate(entity_list):
            entity_span_list.append(i)
            # if ix != tf.cast(0, dtype=tf.int64):
            #     entity_span_list.append(i)
        entity_span_list = entity_span_list[:10]
        relations_entity = []
        relation_masks = []
        relations_num = 0
        for i in entity_span_list:
            ss, se = input_entity_start_end[i]
            for j in entity_span_list:
                os, oe = input_entity_start_end[j]
                if i == j:
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

        relations_entity, relation_mask, relation_nums = self.filter_span(entity_clf, input_entity_start_end, input_max_len)
        if relations_entity.shape[0]:
            relation_feature = build_relation_feature(h, entity_spans_pool, relations_entity, size_embeddings, relation_mask,
                                                      relation_nums)
            rel_clf = self.rel_classifier(relation_feature)
            rel_clf_argmax = tf.argmax(rel_clf, axis=-1)

            rel_res = [(relations_entity[i].numpy()[0], relations_entity[i].numpy()[1], label) for i, label in enumerate(rel_clf_argmax.numpy())]
            rel_res = [(input_entity_start_end[si], entity_list[si], input_entity_start_end[oi], entity_list[oi], p) for si, oi, p in rel_res if p]
            return rel_res
        else:
            return []



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

epoch = 100
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


spert.load_weights(model_path)
batch_data_iter = get_test_sample_data(1)
submit_res = []
for i, batch_data in enumerate(batch_data_iter):
    pres = spert.predict(batch_data["encodings"],
                               batch_data["context_masks"],
                               batch_data["entity_masks"],
                               batch_data["entity_sizes"],
                               batch_data["entity_num"],
                               batch_data["entity_start_end"],
                               batch_data["max_len"])
    doc = data_loader.test_documents[i]
    i_text = doc.text
    spo_list = []
    for sop in pres:
        sub_i, sub_j = sop[0]
        sub_type = sop[1]
        obj_i, obj_j = sop[2]
        obj_type = sop[3]
        pre_type = sop[4]
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
    submit_res.append(single_spo)
    break





