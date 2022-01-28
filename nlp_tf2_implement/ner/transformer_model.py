#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/23 22:59
    @Author  : jack.li
    @Site    : 
    @File    : transformer_model.py

"""

import tensorflow as tf
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import metrix
from nlp_applications.layers.neo_tf2_transformer import TransformerEncoder

msra_data = LoadMsraDataV2("D:\data\\nlp\\命名实体识别\\msra_ner_token_level\\")

char2id = {"pad": 0, "unk": 1}
max_len = -1
msra_train_id = []
msra_pos_id = []
for sentence in msra_data.train_sentence_list:
    sentence_id = []
    for s in sentence:
        if s not in char2id:
            char2id[s] = len(char2id)
        sentence_id.append(char2id[s])
    if len(sentence_id) > max_len:
        max_len = len(sentence_id)
    msra_train_id.append(sentence_id)
    msra_pos_id.append([i+1 for i in range(len(sentence_id))])

tag_list = msra_data.train_tag_list
label2id = {"O": 0}
for lb in msra_data.labels:
    if lb not in label2id:
        label2id[lb] = len(label2id)
id2label = {v:k for k, v in label2id.items()}
msra_tag_id = []
for tag in tag_list:
    tag_ids = []
    for tg in tag:
        tag_ids.append(label2id[tg])
    msra_tag_id.append(tag_ids)

vocab_size = len(char2id)+1
embed_size = 64
class_num = len(label2id)

train_data = tf.keras.preprocessing.sequence.pad_sequences(msra_train_id, padding="post", maxlen=max_len)
label_data = tf.keras.preprocessing.sequence.pad_sequences(msra_tag_id, padding="post", maxlen=max_len)
pos_data = tf.keras.preprocessing.sequence.pad_sequences(msra_pos_id, padding="post", maxlen=max_len)
dataset = tf.data.Dataset.from_tensor_slices((train_data, pos_data, label_data)).shuffle(100).batch(100)

# vocab_size = 10
# embed_size = 64
# data_maxlen = 4
# class_num = 10


class TransformersModel(tf.keras.models.Model):

    def __init__(self, encoder_num, seq_num, header_num=8):
        super(TransformersModel, self).__init__()

        self.embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.position_embed = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.transformer_encoder_list = [TransformerEncoder(embed_size, seq_num, header_num) for _ in range(encoder_num)]

        self.linear = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs, position_inputs, training=None, mask=None):
        embed = self.embed(inputs)
        posi_embed = self.position_embed(position_inputs)

        embed_value = embed + posi_embed
        for encoder in self.transformer_encoder_list:
            embed_value = encoder(embed_value)

        ner_logits = self.linear(embed_value)

        return ner_logits




def run_test_model():
    model = TransformersModel(2, 4, 2)
    sample_input = tf.constant([[1, 3, 3, 2]])
    sample_position = tf.constant([[1, 2, 3, 4]])

    print(model(sample_input, sample_position))


# run_test_model()

model = TransformersModel(2, max_len, 2)
optimizer = tf.keras.optimizers.Adam()


def loss_func(input_y, logits):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(input_y, 0))

    mask = tf.cast(mask, dtype=tf.int64)
    lossv = cross_func(input_y, logits, sample_weight=mask)

    return lossv


@tf.function()
def train_step(input_xx, input_position, input_yy):

    with tf.GradientTape() as tape:
        logits = model(input_xx, input_position)
        loss_v = loss_func(input_yy, logits)

    variables = model.variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v


epoch = 5
for ep in range(epoch):

    for batch, (trainv, posv, labelv) in enumerate(dataset.take(-1)):
        loss = train_step(trainv, posv, labelv)

        if batch % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss))
    break
