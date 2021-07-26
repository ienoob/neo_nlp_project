#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import argparse
import numpy as np
import torch
import jieba
import torch.nn as nn
from torch.optim import Optimizer
from transformers import BertModel
from pytorch.layers.crf import CRF
import torch.autograd as autograd
import torch.optim as optim
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertTokenizer
from nlp_applications.data_loader import LoadMsraDataV2
from pytorch.ner.latticelstm import LatticeLSTMModel
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3

msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")
bert_model_name = "bert-base-chinese"
class_num = len(msra_data.label2id)
# id2label = {v: k for k, v in msra_data.label2id.items()}


def sequence_padding(inputs, length=None, padding=0, is_float=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return torch.tensor(out_tensor)


class DataIterator(object):
    def __init__(self, input_loader, input_batch_num):
        self.input_loader = input_loader
        self.input_batch_num = input_batch_num
        self.entity_label2id = {"O": 0}
        self.max_len = 0
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.char2id = {"pad": 0, "<unk>": 1}
        self.gz2id = {"pad": 0, "<unk>": 1}
        self.char_freq = dict()
        for sentence in self.input_loader.train_sentence_list:
            for char in sentence:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                self.char_freq.setdefault(char, 0)
                self.char_freq[char] += 1

            for word in jieba.cut("".join(sentence)):
                if len(word) < 2:
                    continue
                if word not in self.gz2id:
                    self.gz2id[word] = len(self.gz2id)

        # 将少数char 作为unk 样本
        for k, v in self.char_freq.items():
            if v == 1:
                self.char2id[k] = self.char2id["<unk>"]

        for tag in self.input_loader.labels:
            if tag not in self.entity_label2id:
                self.entity_label2id[tag] = len(self.entity_label2id)

    def _transformer2feature(self, sentence, label_list):
        sentence_id = [self.char2id.get(char, self.char2id["<unk>"]) for char in sentence]
        label_id = [self.input_loader.label2id[label_i] for label_i in label_list]
        # sentence_bert_mask = sentence_mask + [1, 1]
        sentence_d = dict()
        idx = 0
        for word in jieba.cut("".join(sentence)):
            if word not in self.gz2id:
                continue
            sentence_d[idx] = word
            idx += len(word)

        gaz_list = []
        for iv in range(len(sentence)):
            if iv in sentence_d:
                word = sentence_d[iv]

                gaz_list.append([[self.gz2id[word]],[len(word)]])
            else:
                gaz_list.append([])

        return {
            "gaz_list": gaz_list,
            "sentence_id": sentence_id,
            "label_id": label_id,
        }

    def batch_transformer(self, input_batch_data):
        batch_sentence_id = []
        batch_label_id = []
        batch_gaz_list = []
        max_len = 0
        for data in input_batch_data:
            batch_sentence_id.append(data["sentence_id"])
            batch_label_id.append(data["label_id"])
            max_len = max(max_len, len(data["sentence_id"]))
            batch_gaz_list.append(data["gaz_list"])
        batch_sentence_id = autograd.Variable(sequence_padding(batch_sentence_id, length=max_len)).long()
        batch_label_id = autograd.Variable(sequence_padding(batch_label_id, length=max_len)).long()

        return {
            "sentence_id": batch_sentence_id,
            "label_id": batch_label_id,
            "gaz_list": batch_gaz_list
        }

    def __iter__(self):
        inner_batch_data = []
        for i, sentence in enumerate(self.input_loader.train_sentence_list):
            tf_data = self._transformer2feature(sentence, self.input_loader.train_tag_list[i])
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        if inner_batch_data:
            yield self.batch_transformer(inner_batch_data)

    def iter_test(self):
        inner_batch_data = []
        for i, sentence in enumerate(self.input_loader.test_sentence_list):
            tf_data = self._transformer2feature(sentence, self.input_loader.test_tag_list[i])
            inner_batch_data.append(tf_data)
            if len(inner_batch_data) == self.input_batch_num:
                yield self.batch_transformer(inner_batch_data)
                inner_batch_data = []
        if inner_batch_data:
            yield self.batch_transformer(inner_batch_data)

def evaluation(model, data_iterator, id2label):
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0
    for idx, batch_data in enumerate(data_iterator.iter_test()):
        try:
            tag_seqs = model(batch_data["gaz_list"], batch_data["sentence_id"])
        except Exception:
            print(len(batch_data["gaz_list"][0]))
            print(len(batch_data["sentence_id"][0]))
            # print(batch_data["gaz_list"], batch_data["sentence_id"])
            raise Exception
        tag_seqs = tag_seqs.numpy()

        # print(tag_seqs.shape)
        label_seq = batch_data["label_id"].numpy()
        batch_num_value = label_seq.shape[0]

        for b in range(batch_num_value):
            tag_seq_list = tag_seqs[b]
            tag_seq_list = [id2label.get(tag, "O") for tag in tag_seq_list]
            # print(tag_seq_list)

            true_seq_list = label_seq[b]
            true_seq_list = [id2label.get(tag, "O") for tag in true_seq_list]
            # print(true_seq_list)

            pre_value = extract_entity(tag_seq_list)
            true_value = extract_entity(true_seq_list)
            # print(true_value)

            pre_num += len(pre_value)
            true_num += len(true_value)

            for p in pre_value:
                if p in true_value:
                    hit_num += 1
    return {
        "hit_num": hit_num,
        "true_num": true_num,
        "pre_num": pre_num
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument("--lr_decay", default=0.0005)
    parser.add_argument("--lr", default=0.0005)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--warm_up", default=False)

    args = parser.parse_args()

    iteration = 10
    train_Ids = []
    data_iterator = DataIterator(msra_data, args.batch_size)
    id2label = {v: k for k, v in data_iterator.entity_label2id.items()}

    char_size = len(data_iterator.char2id)
    gz_size = len(data_iterator.gz2id)
    char_embed_size = 64
    lstm_hidden = 64
    label_size = len(data_iterator.entity_label2id)

    model = LatticeLSTMModel(char_size, char_embed_size, lstm_hidden, label_alphabet_size=label_size, gaz_alphabet_size=gz_size)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters)

    for iter in range(iteration):

        model.train()
        model.zero_grad()

        for idx, batch_data in enumerate(data_iterator):
            loss, tag_seq = model.neg_log_likelihood_loss(batch_data["gaz_list"], batch_data["sentence_id"],
                                                          batch_data["label_id"])
            # v = model(batch_data["gaz_list"], batch_data["sentence_id"])

            if idx % 100 == 0:
                print("epoch {0} batch {1} loss value {2}".format(iter, idx, loss.data.numpy()))

            if idx and idx % 500 == 0:
                eval_res = evaluation(model, data_iterator, id2label)
                print(eval_metrix_v3(eval_res["hit_num"], eval_res["true_num"], eval_res["pre_num"]))

            loss.backward()
            optimizer.step()
            model.zero_grad()
        break
