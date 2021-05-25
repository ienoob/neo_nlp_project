#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/2/20 17:59
    @Author  : jack.li
    @Site    : 
    @File    : rnn_crf_model.py

"""
"""
    鉴于单纯的rnn模型不行，这里试着增加crf层
"""
import tensorflow as tf
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import metrix
from nlp_applications.ner.crf_addons import crf_log_likelihood, viterbi_decode


class TF2CRF(tf.keras.layers.Layer):

    def __init__(self, num_tags, batch_first):
        super(TF2CRF, self).__init__()

        # self.average_batch = False
        # self.tagset_size = tagset_size
        #
        # init_transitions = tf.zeros(self.tagset+2, self.tagset_size+2)
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = tf.keras.layers.Dense(num_tags)
        self.end_transitions = tf.keras.layers.Dense(num_tags)
        self.transitions = tf.keras.layers.Dense(num_tags)

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

word_num = len(char2id)+1
embed_size = 64
rnn_dim = 10
class_num = len(label2id)

train_data = tf.keras.preprocessing.sequence.pad_sequences(msra_train_id, padding="post", maxlen=max_len)
label_data = tf.keras.preprocessing.sequence.pad_sequences(msra_tag_id, padding="post", maxlen=max_len)
dataset = tf.data.Dataset.from_tensor_slices((train_data, label_data)).shuffle(100).batch(100)


char_size = len(char2id)+1
embedding_size = 64
lstm_dim = 10
label_num = len(label2id)


class LSTMCRF(tf.keras.Model):

    def __init__(self):
        super(LSTMCRF, self).__init__()

        self.embedding = tf.keras.layers.Embedding(char_size, embedding_size)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_num)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_num, label_num)))
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, input_text, labels=None,  training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(input_text, 0), dtype=tf.int32), axis=-1)
        mask = tf.math.logical_not(tf.math.equal(input_text, 0))
        inputs = self.embedding(input_text)
        inputs = self.bi_lstm(inputs, mask=mask)
        inputs = self.dropout(inputs, training)
        logits = self.dense(inputs)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = crf_log_likelihood(logits, label_sequences, text_lens,self.transition_params)

            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens



model = LSTMCRF()
input_x_sample = tf.constant([[1, 2]])
input_y_sample = tf.constant([[1, 2]])
# output_y, _,  = model(input_x_sample, True)

optimizer = tf.keras.optimizers.Adam()


def loss_func(input_y, logits):
    cross_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(input_y, 0))

    mask = tf.cast(mask, dtype=tf.int64)
    lossv = cross_func(input_y, logits, sample_weight=mask)

    return lossv


# print(loss_func(input_x_sample, output_y))


@tf.function()
def train_step(input_xx, input_yy):

    with tf.GradientTape() as tape:
        logits, text_len, log_likelihood = model(input_xx, input_yy, True)
        loss_v = -tf.reduce_mean(log_likelihood)
        loss_v += loss_func(input_yy, logits)

    variables = model.variables
    gradients = tape.gradient(loss_v, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss_v, logits, text_len

def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = viterbi_decode(logit[:text_len], model.transition_params)
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

output_dir = "D:\\tmp\\neo_nlp\\rnn_crf"
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          output_dir,
                                          checkpoint_name='model.ckpt',
                                          max_to_keep=3)
epoch = 20
for ep in range(epoch):

    for batch, (trainv, labelv) in enumerate(dataset.take(-1)):
        loss, logits, text_lens = train_step(trainv, labelv)

        if batch % 10 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labelv)
            print("epoch {0} batch {1} loss is {2} accuracy is {3}".format(ep, batch, loss, accuracy))
            ckpt_manager.save()



ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
ckpt.restore(tf.train.latest_checkpoint(output_dir))

def predict(input_s_list):
    max_v_len = max([len(input_s) for input_s in input_s_list])
    dataset = tf.keras.preprocessing.sequence.pad_sequences([[char2id.get(char, 0) for char in input_str] for input_str in input_s_list], padding='post', maxlen=max_v_len)
    logits, text_lens = model.predict(dataset)
    paths = []
    for logit, text_len in zip(logits, text_lens):
        viterbi_path, _ = viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)

    output_label = []
    for i, output_id in enumerate(paths):
        olen = len(output_id)
        ilen = len(input_s_list[i])
        if olen < ilen:
            output_label.append([id2label[o] for o in output_id]+["O"]*(ilen-olen))
        else:
            output_label.append([id2label[o] for o in output_id][:ilen])
    # output_label = [[id2label[o] for o in output_id] for i, output_id in
    #                 enumerate(paths)]

    print(output_label[0])

    return output_label
    # print([id2tag[id] for id in paths[0]])
    #
    # entities_result = format_result(list(text), [id2tag[id] for id in paths[0]])
    # print(json.dumps(entities_result, indent=4, ensure_ascii=False))

predict(["1月18日，在印度东北部一座村庄，一头小象和家人走过伐木工人正在清理的区域时被一根圆木难住了。"])
predict_labels = predict(msra_data.test_sentence_list)
true_labels = msra_data.test_tag_list

print(metrix(true_labels, predict_labels))

"""
crf 层一加，效果比较明显的提高
    训练50epoch 
    (0.6959573953038005, 0.8118029083721587) 
"""