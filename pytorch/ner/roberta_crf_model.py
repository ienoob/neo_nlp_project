#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import RobertaModel, BertModel
from pytorch.layers.crf import CRF
import torch.autograd as autograd
import torch.optim as optim
from pytorch.layers.bert_optimization import BertAdam
from transformers import RobertaTokenizer, BertTokenizer
from nlp_applications.data_loader import LoadMsraDataV2
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3

msra_data = LoadMsraDataV2("D:\data\\ner\\msra_ner_token_level\\")
bert_model_name = "hfl/chinese-roberta-wwm-ext"
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
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def _transformer2feature(self, sentence, label_list):
        sentence_id = self.tokenizer.encode(sentence)
        label_id = [self.input_loader.label2id[label_i] for label_i in label_list]
        sentence_bert_mask = [1 for _ in sentence_id]
        sentence_mask = np.ones(len(sentence_id)-1)
        # sentence_bert_mask = sentence_mask + [1, 1]

        return {
            "sentence_id": sentence_id,
            "label_id": label_id,
            "sentence_mask": sentence_mask,
            "sentence_bert_mask": sentence_bert_mask
        }

    def batch_transformer(self, input_batch_data):
        batch_sentence_id = []
        batch_label_id = []
        batch_sentence_mask = []
        batch_bert_sentence_mask = []
        max_len = 0
        for data in input_batch_data:
            batch_sentence_id.append(data["sentence_id"])
            batch_label_id.append(data["label_id"])
            batch_sentence_mask.append(data["sentence_mask"])
            batch_bert_sentence_mask.append(data["sentence_bert_mask"])
            max_len = max(max_len, len(data["sentence_id"]))
        max_len = min(512, max_len)
        batch_sentence_id = autograd.Variable(sequence_padding(batch_sentence_id, length=max_len)).long()
        batch_label_id = autograd.Variable(sequence_padding(batch_label_id, length=max_len-2)).long()
        batch_bert_sentence_mask = sequence_padding(batch_bert_sentence_mask, length=max_len)
        batch_bert_sentence_mask = autograd.Variable(batch_bert_sentence_mask).byte()
        batch_sentence_mask = sequence_padding(batch_sentence_mask, length=max_len-2)
        batch_sentence_mask = autograd.Variable(batch_sentence_mask).byte()

        return {
            "sentence_id": batch_sentence_id,
            "label_id": batch_label_id,
            "sentence_mask": batch_sentence_mask,
            "bert_sentence_mask": batch_bert_sentence_mask
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


class RobertaModelNER(nn.Module):
    def __init__(self, gpu_use=False, bertpath=None, label_alphabet_size=2, dropout=0.5):
        super(RobertaModelNER, self).__init__()

        self.gpu = gpu_use
        self.data_bertpath = bertpath
        self.bertpath = self.data_bertpath

        char_feature_dim = 768
        print('total char_feature_dim is {}'.format(char_feature_dim))

        self.bert_encoder = BertModel.from_pretrained(self.bertpath)

        self.hidden2tag = nn.Linear(char_feature_dim, label_alphabet_size + 2)
        self.drop = nn.Dropout(p=dropout)

        self.crf = CRF(label_alphabet_size, self.gpu)

        if self.gpu:
            self.bert_encoder = self.bert_encoder.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.crf = self.crf.cuda()

    def get_tags(self, batch_bert, bert_mask):
        seg_id = torch.zeros(bert_mask.size()).long().cuda() if self.gpu else torch.zeros(bert_mask.size()).long()
        outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
        outputs = outputs[0][:, 1:-1, :]
        tags = self.hidden2tag(outputs)

        return tags

    def neg_log_likelihood_loss(self, mask, batch_label, batch_bert,
                                bert_mask):
        tags = self.get_tags(batch_bert, bert_mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, mask, batch_bert, bert_mask):
        tags = self.get_tags(batch_bert, bert_mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq


MAX_SENTENCE_LENGTH = 512


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()

    return right_token, total_token


def evaluation(model, data_iterator):
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0
    for idx, batch_data in enumerate(data_iterator.iter_test()):
        tag_seqs = model(batch_data["sentence_mask"].byte(),
                         batch_data["sentence_id"],
                         batch_data["bert_sentence_mask"].byte())
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
    metric = eval_metrix_v3(hit_num, true_num, pre_num)
    return {
        "hit_num": hit_num,
        "true_num": true_num,
        "pre_num": pre_num,
        "recall": metric["recall"],
        "precision": metric["precision"],
        "f1_value": metric["f1_value"]
    }

def check_test_label():
    data_iterator = DataIterator(msra_data, 100)
    for idx, batch_data in enumerate(data_iterator.iter_test()):
        label_seq = batch_data["label_id"].numpy()
        batch_num_value = label_seq.shape[0]

        for b in range(batch_num_value):
            true_seq_list = label_seq[b]
            true_seq_list = [id2label.get(tag, "O") for tag in true_seq_list]
            true_value = extract_entity(true_seq_list)
            print(true_value)


if __name__ == "__main__":
    input_file = ""
    bertpath = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', help='Embedding for words', default='None')
    parser.add_argument('--status', choices=['train', 'test'], help='update algorithm', default='train')
    parser.add_argument("--lr_decay",  default=0.0005)
    parser.add_argument("--lr", default=0.00005)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--warm_up", default=False)

    args = parser.parse_args()

    iteration = 10
    train_Ids = []
    data_iterator = DataIterator(msra_data, args.batch_size)

    model = RobertaModelNER(bertpath=bert_model_name, label_alphabet_size=class_num)
    # parameter = bert_model.parameters()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters)
    if args.warm_up:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = int(len(train_Ids) / args.batch_size) * iteration
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=0.1,
                             t_total=num_train_optimization_steps)

    for iter in range(iteration):
        for idx, batch_data in enumerate(data_iterator):

            # epoch_start = time.time()
            # temp_start = epoch_start
            # print(("Epoch: %s/%s" % (idx, HP_iteration)))
            if not args.warm_up:
                optimizer = lr_decay(optimizer, idx, args.lr_decay, args.lr)

            model.train()
            model.zero_grad()

            loss, tag_seq = model.neg_log_likelihood_loss(batch_data["sentence_mask"].byte(),
                                                          batch_data["label_id"],
                                                          batch_data["sentence_id"],
                                                          batch_data["bert_sentence_mask"].byte())
            loss_value = loss.data.cpu().numpy()
            right, whole = predict_check(tag_seq, batch_data["label_id"], batch_data["sentence_mask"])
            print(right, whole, idx)
            print("epoch {0} batch {1} loss value is {2}".format(iter, idx, loss_value))

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if idx and idx % 100 == 0:
                print(evaluation(model, data_iterator))
    # check_test_label()
    # print(len(id2label))


        # print(tag_seqs.shape)
        # max_v = torch.max(tag_seqs)
        # print(max_v)
        # tag_seq_list = [[id2label.get(tag, "O") for tag in tag_seq] for tag_seq in tag_seqs.numpy() ]
        # print(tag_seq_list[0])
        # # print(tag_seq_list)
        # #
        # print(batch_data["label_id"])
