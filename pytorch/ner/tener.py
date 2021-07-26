#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
from torch import nn
from pytorch.layers.crf import CRF
from pytorch.layers.transformer import TransformerEncoder

import argparse
import numpy as np
import torch
from torch.optim import Optimizer
from transformers import BertModel
from pytorch.layers.crf import CRF
import torch.autograd as autograd
import torch.optim as optim
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertTokenizer
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3


msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")
bert_model_name = "bert-base-chinese"
class_num = len(msra_data.label2id)
id2label = {v: k for k, v in msra_data.label2id.items()}


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
        self.char_freq = dict()
        for sentence in self.input_loader.train_sentence_list:
            for char in sentence:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                self.char_freq.setdefault(char, 0)
                self.char_freq[char] += 1

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

        return {
            "sentence_id": sentence_id,
            "label_id": label_id,
        }

    def batch_transformer(self, input_batch_data):
        batch_sentence_id = []
        batch_label_id = []
        max_len = 0
        for data in input_batch_data:
            batch_sentence_id.append(data["sentence_id"])
            batch_label_id.append(data["label_id"])
            max_len = max(max_len, len(data["sentence_id"]))
        max_len = min(512, max_len)
        batch_sentence_id = autograd.Variable(sequence_padding(batch_sentence_id, length=max_len)).long()
        batch_label_id = autograd.Variable(sequence_padding(batch_label_id, length=max_len)).long()

        return {
            "sentence_id": batch_sentence_id,
            "label_id": batch_label_id,
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


class TENER(nn.Module):

    def __init__(self, char_size, embed_size, d_model, num_layers, n_head, feedforward_dim, dropout, tag_vocab_num,
                 after_norm=True, attn_type='adatrans', bi_embed=None,bert_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):
        super(TENER, self).__init__()

        self.embed = nn.Embedding(char_size, embed_size)

        self.in_fc = nn.Linear(embed_size, d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)

        self.crf = CRF(tag_vocab_num, False)
        self.out_fc = nn.Linear(d_model, tag_vocab_num+2)

    def forward(self, input_chars, batch_label=None):
        mask = input_chars.ne(0)
        chars = self.embed(input_chars)
        chars = self.in_fc(chars)
        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)
        tags = self.out_fc(chars)

        if batch_label is None:
            scores, tag_seq = self.crf._viterbi_decode(tags, mask)
            return tag_seq
        else:
            total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(tags, mask)

            return total_loss, tag_seq


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    # print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluation(model, data_iterator):
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0
    for idx, batch_data in enumerate(data_iterator.iter_test()):
        tag_seqs = model(batch_data["sentence_id"])
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
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--warm_up", default=True)

    args = parser.parse_args()

    iteration = 100
    train_Ids = []
    data_iterator = DataIterator(msra_data, args.batch_size)
    char_size = len(data_iterator.char2id)
    embed_size = 128
    d_model = 128
    num_layers = 2
    n_head = 8
    feedforward_dim = 128
    dropout = 0.3
    tag_vocab_num = len(data_iterator.entity_label2id)

    model = TENER(char_size, embed_size, d_model, num_layers, n_head, feedforward_dim, dropout, tag_vocab_num)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters)

    for iter in range(iteration):

        model.train()
        model.zero_grad()

        for idx, batch_data in enumerate(data_iterator):

            # epoch_start = time.time()
            # temp_start = epoch_start
            # print(("Epoch: %s/%s" % (idx, HP_iteration)))
            if args.warm_up:
                optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)

            loss, tag_seq = model.forward(batch_data["sentence_id"], batch_data["label_id"])

            if idx % 100 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value {2}".format(iter, idx, loss_value))

            if idx % 500 == 0:
                eval_res = evaluation(model, data_iterator)
                print(eval_metrix_v3(eval_res["hit_num"], eval_res["true_num"], eval_res["pre_num"]))

            loss.backward()
            optimizer.step()
            model.zero_grad()
