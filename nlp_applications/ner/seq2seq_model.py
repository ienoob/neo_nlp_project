#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/3/8 22:09
    @Author  : jack.li
    @Site    : 
    @File    : seq2seq_model.py

"""
import tensorflow as tf
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import metrix_v2



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
label2id = {"O": 0, "B-ST": 1, "B-ED": 2}

for lb in msra_data.labels:
    if lb not in label2id:
        label2id[lb] = len(label2id)

id2label = {v:k for k, v in label2id.items()}
msra_tag_id_xy = []
msra_tag_id_yy = []
for tag in tag_list:
    tag_ids_xy = [label2id["B-ST"]]
    tag_ids_yy = []
    for tg in tag:
        tag_ids_xy.append(label2id[tg])
        tag_ids_yy.append(label2id[tg])
    tag_ids_yy.append(label2id["B-ED"])
    msra_tag_id_xy.append(tag_ids_xy)
    msra_tag_id_yy.append(tag_ids_yy)

train_data = tf.keras.preprocessing.sequence.pad_sequences(msra_train_id, padding="post", maxlen=max_len)
input_xy_data = tf.keras.preprocessing.sequence.pad_sequences(msra_tag_id_xy, padding="post", maxlen=max_len)
input_yy_data = tf.keras.preprocessing.sequence.pad_sequences(msra_tag_id_yy, padding="post", maxlen=max_len)
dataset = tf.data.Dataset.from_tensor_slices((train_data, input_xy_data, input_yy_data)).shuffle(100).batch(100)

encoder_char_size = len(char2id)+1
encoder_embed = 64
encoder_lstm_embed = 64

decoder_char_size = len(label2id)
decoder_embed = 64
decoder_lstm_embed = 64

class_num = len(label2id)


class Encoder(tf.keras.models.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embed = tf.keras.layers.Embedding(encoder_char_size, encoder_embed)
        self.lstm = tf.keras.layers.LSTM(encoder_lstm_embed, return_state=True)

    def call(self, inputs, training=None, mask=None):
        x = self.embed(inputs)
        mask = tf.math.logical_not(tf.math.equal(inputs, 0))
        x, c_state, s_state = self.lstm(x, mask=mask)

        return x, (c_state, s_state), mask


class Decoder(tf.keras.models.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.embed = tf.keras.layers.Embedding(decoder_char_size, decoder_embed)
        self.lstm = tf.keras.layers.LSTM(decoder_lstm_embed, return_state=True, return_sequences=True)

        self.out = tf.keras.layers.Dense(class_num, activation="softmax")

    def call(self, inputs, training=None, mask=None, input_state=None):
        x = self.embed(inputs)
        x, c_state, s_state = self.lstm(x, initial_state=input_state)

        out = self.out(x)

        return out, (c_state, s_state)


optimizer = tf.keras.optimizers.Adam()


def loss_func(input_y, logits, input_mask):
    _loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

    mask = tf.cast(input_mask, dtype=tf.int64)
    lossv = _loss_func(input_y, logits, sample_weight=mask)

    return lossv


encoder = Encoder()
decoder = Decoder()


@tf.function()
def train_step(input_xx, input_xy, input_yy):

    with tf.GradientTape() as tape:
        _, encoder_state, e_mask = encoder(input_xx)
        decoder_out, _ = decoder(input_xy, input_state=encoder_state)

        loss_v = loss_func(input_yy, decoder_out, e_mask)

    variables = encoder.variables + decoder.variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v


test_input_xx = tf.constant([[1, 3]])
test_input_xy = tf.constant([[1, 2]])
test_input_yy = tf.constant([[1, 2]])


epoch = 20
for ep in range(epoch):
    for batch, (input_xx, input_xy, input_yy) in enumerate(dataset.take(-1)):
        loss = train_step(input_xx, input_xy, input_yy)

        if batch % 10 == 0:
            print("epoch {0} batch {1} loss is {2}".format(ep, batch, loss))



def predict(input_s_list):
    batch_num = 100
    b_num = int(len(input_s_list)//batch_num) + 1
    predict_res = []
    for b in range(b_num):
        sub_input_s_list = input_s_list[b*batch_num:b*batch_num+batch_num]
        if len(sub_input_s_list) == 0:
            break
        input_s_id = [[char2id.get(s, 1) for s in input_s] for input_s in sub_input_s_list]
        max_v_len = max([len(input_s) for input_s in sub_input_s_list])
        input_s_id = tf.keras.preprocessing.sequence.pad_sequences(input_s_id, padding="post", maxlen=max_v_len)
        _, encoder_state, _ = encoder(input_s_id)

        input_decoder_state = encoder_state
        input_xy = [[label2id["B-ST"]]]*len(sub_input_s_list)
        input_xy = tf.cast(input_xy, dtype=tf.int64)

        input_out = None
        for i in range(max_v_len):
            decoder_out, input_decoder_state = decoder(input_xy, input_state=input_decoder_state)
            decoder_out_fi = decoder_out[:,-1,:]
            decoder_out_logits = tf.argmax(decoder_out_fi, axis=-1)
            decoder_out_logits = tf.reshape(decoder_out_logits, (len(sub_input_s_list), -1))
            decoder_out_logits =  tf.cast(decoder_out_logits, dtype=tf.int64)
            if input_out is None:
                input_out = decoder_out_logits
            else:
                input_out = tf.concat([input_out, decoder_out_logits], 1)
            input_xy = tf.concat([input_xy, decoder_out_logits], 1)

        output_label = [[id2label[o] for o in output_id] for i, output_id in
                        enumerate(input_out.numpy())]
        predict_res += output_label

    npredict_res = []
    for i, pres in enumerate(predict_res):
        npredict_res.append(pres[:len(input_s_list[i])])

    return npredict_res


out_label = predict(["1月18日，在印度东北部一座村庄，一头小象和家人走过伐木工人正在清理的区域时被一根圆木难住了。"])
print(out_label)

predict_labels = predict(msra_data.test_sentence_list)
# for test_sentence in msra_data.test_sentence_list:
#     predict_label = predict([test_sentence])
#     predict_labels.append(predict_label[0])
# predict_labels = predict(msra_data.test_sentence_list)
true_labels = msra_data.test_tag_list

print(metrix_v2(true_labels, predict_labels))

"""
    {'score': 311, 'p_count': 1677, 'd_count': 8396, 'recall': 0.037041443898112925, 
    'precision': 0.1854500981275503, 'f1_value': 0.0617492183581419} ╮(╯▽╰)╭ seq2seq 确实不适合做ner 任务，
"""








