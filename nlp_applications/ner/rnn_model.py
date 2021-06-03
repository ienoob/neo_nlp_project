#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/2/20 17:58
    @Author  : jack.li
    @Site    : 
    @File    : rnn_model.py

"""
import os
import tensorflow as tf
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import metrix


msra_data = LoadMsraDataV2("D:\data\\nlp\\命名实体识别\\msra_ner_token_level\\")


char2id = {"pad": 0, "unk": 1}
max_len = -1
msra_train_id = []
for sentence in msra_data.train_sentence_list:
    sentence_id = []
    for s in sentence:
        if s not in char2id:
            char2id[s] = len(char2id)
        sentence_id.append(char2id[s])
    if len(sentence_id) > max_len:
        max_len = len(sentence_id)
    msra_train_id.append(sentence_id)

tag_list = msra_data.train_tag_list
label2id = {"pad": 0, "O": 1}
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

word_num = len(char2id)+1
embed_size = 64
rnn_dim = 10
class_num = len(label2id)

train_data = tf.keras.preprocessing.sequence.pad_sequences(msra_train_id, padding="post", maxlen=max_len)
label_data = tf.keras.preprocessing.sequence.pad_sequences(msra_tag_id, padding="post", maxlen=max_len)
dataset = tf.data.Dataset.from_tensor_slices((train_data, label_data)).shuffle(100).batch(100)


class RNNNER(tf.keras.Model):

    def __init__(self):
        super(RNNNER, self).__init__()
        self.embed = tf.keras.layers.Embedding(word_num, output_dim=embed_size)
        # self.rnn = tf.keras.layers.SimpleRNN(rnn_dim, return_sequences=True, return_state=True)
        # self.rnn = tf.keras.layers.LSTM(rnn_dim, return_sequences=True, return_state=True)
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True))
        # self.dropout = tf.keras.layers.Dropout(0.5)
        self.out = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, input_x, training=False, input_state=None):
        mask = tf.math.logical_not(tf.math.equal(input_x, 0))
        input_x = self.embed(input_x)

        input_x = self.rnn(input_x, mask=mask)
        # input_x = self.dropout(input_x, training)
        input_x = self.out(input_x)

        return input_x


rnn = RNNNER()


optimizer = tf.keras.optimizers.Adam()

def loss_func(input_y, logits):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(input_y, 0))

    mask = tf.cast(mask, dtype=tf.int64)
    lossv = cross_func(input_y, logits, sample_weight=mask)

    return lossv


def run_test_model():
    input_x_sample = tf.constant([[1, 2]])
    input_y_sample = tf.constant([[1, 2]])
    output_y = rnn(input_x_sample, True)
    print(loss_func(input_y_sample, output_y))


@tf.function()
def train_step(input_xx, input_yy):

    with tf.GradientTape() as tape:
        logits = rnn(input_xx, True)
        loss_v = loss_func(input_yy, logits)

    variables = rnn.variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v

# dv = tf.constant([[[1,2],[2,3]]])
# print(tf.argmax(dv, 2))


rnn_data_path = "D:\\tmp\\neo_nlp\\rnn_ner"
if os.path.exists(rnn_data_path):
    rnn.load_weights(rnn_data_path)
# else:
epoch = 1
for ep in range(epoch):

    for batch, (trainv, labelv) in enumerate(dataset.take(-1)):
        loss_value = train_step(trainv, labelv)

        if batch % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss_value))
            # rnn.save(rnn_data_path)
            rnn.save_weights(rnn_data_path, save_format='tf')


def predict(input_s_listv):
    batch_deal_num = 100
    b_num = int(len(input_s_listv) // batch_deal_num) + 1
    output_label = []
    for bi in range(b_num):
        input_s_list = input_s_listv[bi*batch_deal_num:bi*batch_deal_num+batch_deal_num]
        if len(input_s_list) == 0:
            continue
        input_s_id = [[char2id.get(s, 1) for s in input_s] for input_s in input_s_list]
        max_v_len = max([len(input_s) for input_s in input_s_list])
        input_s_id = tf.keras.preprocessing.sequence.pad_sequences(input_s_id, padding="post", maxlen=max_v_len)
        # input_s_id = tf.cast(input_s_id, dtype=tf.int64)
        output_ids = rnn(input_s_id)

        output_idv = tf.argmax(output_ids, axis=-1)

        output_label += [[id2label[o] for o in output_id][:len(input_s_list[i])] for i, output_id in enumerate(output_idv.numpy())]

    # print(output_label)

    return output_label


out_label = predict(["1月18日，在印度东北部一座村庄，一头小象和家人走过伐木工人正在清理的区域时被一根圆木难住了。"])
print(out_label)
predict_labels = predict(msra_data.test_sentence_list)
true_labels = msra_data.test_tag_list

print(metrix(true_labels, predict_labels))

"""
单层循环神经网络效果还行吧
(0.6138949406923263, 0.7106627434496287) 15 epoch
"""












