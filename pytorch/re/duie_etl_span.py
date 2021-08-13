#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import jieba
import argparse
import logging
import torch
import time
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import List
from torch import optim
from functools import partial
from torch.utils.data import Dataset, DataLoader
from nlp_applications.data_loader import LoaderDuie2Dataset, Document, BaseDataIterator
from pytorch.re.etl_span import EtlSpan
from nlp_applications.ie_relation_extraction.evaluation import eval_metrix

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

data_path = "D:\data\关系抽取"
data_loader = LoaderDuie2Dataset(data_path)

# class SPOExample(object):

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
    return out_tensor.clone().detach()


class Duie2Dataset(Dataset):
    def __init__(self, documents, char2idx, word2idx, rel_num, is_train=True):
        super(Duie2Dataset, self).__init__()
        self.char2idx = char2idx
        self.word2idx = word2idx
        self.rel_num = rel_num
        self.is_train = is_train
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def _create_collate_fn(self, batch_first=False):
        def collate(documents: List[Document]):
            batch_char_ids, batch_word_ids, batch_sentence_len = [], [], []
            batch_gold_answer = []
            batch_token_type_ids, batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], [], []
            for document in documents:
                char_ids = document.text_id
                word_ids = [self.word2idx.get(word, 0) for word in jieba.cut(document.raw_text) for _ in word]
                batch_sentence_len.append(len(char_ids))

                assert len(char_ids) == len(word_ids)

                if self.is_train:
                    spoes = {}
                    for relation in document.relation_list:
                        s = relation.sub.entity_text
                        p = relation.id
                        o = relation.obj.entity_text
                        s_idx = relation.sub.start
                        o_idx = relation.obj.start
                        if s_idx != -1 and o_idx != -1:
                            s = (s_idx, s_idx + len(s) - 1)
                            o = (o_idx, o_idx + len(o) - 1, p)
                            if s not in spoes:
                                spoes[s] = []
                            spoes[s].append(o)

                    if spoes:
                        # subject标签
                        token_type_ids = np.zeros(len(char_ids), dtype=np.long)
                        subject_labels = np.zeros((len(char_ids), 2), dtype=np.float32)
                        for s in spoes:
                            subject_labels[s[0], 0] = 1
                            subject_labels[s[1], 1] = 1
                        # 随机选一个subject
                        start, end = np.array(list(spoes.keys())).T
                        start = np.random.choice(start)
                        end = np.random.choice(end[end >= start])
                        token_type_ids[start:end + 1] = 1
                        subject_ids = (start, end)
                        # 对应的object标签
                        object_labels = np.zeros((len(char_ids), self.rel_num, 2), dtype=np.float32)
                        for o in spoes.get(subject_ids, []):
                            object_labels[o[0], o[2], 0] = 1
                            object_labels[o[1], o[2], 1] = 1
                        batch_char_ids.append(char_ids)
                        batch_word_ids.append(word_ids)
                        batch_token_type_ids.append(token_type_ids)
                        batch_subject_labels.append(subject_labels)
                        batch_subject_ids.append(subject_ids)
                        batch_object_labels.append(object_labels)


                else:
                    gold_answer = []
                    for relation in document.relation_list:
                        s = relation.sub
                        p = relation.id
                        o = relation.obj

                        gold_answer.append((s.start, s.end-1, o.start, o.end-1, p))

                    batch_char_ids.append(char_ids)
                    batch_word_ids.append(word_ids)
                    batch_gold_answer.append(gold_answer)

            batch_char_ids = sequence_padding(batch_char_ids, is_float=False)
            batch_word_ids = sequence_padding(batch_word_ids, is_float=False)

            if self.is_train:
                batch_token_type_ids = sequence_padding(batch_token_type_ids, is_float=False)
                batch_subject_ids = torch.tensor(batch_subject_ids)
                batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2), is_float=True)
                batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((self.rel_num, 2)),
                                                       is_float=True)

                return batch_char_ids, batch_word_ids, batch_token_type_ids, batch_subject_ids, batch_subject_labels, batch_object_labels, batch_sentence_len
            else:
                return batch_char_ids, batch_word_ids, batch_sentence_len, batch_gold_answer

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def _load_embedding(embedding_file, embedding_dict):

    with open(embedding_file, encoding="utf-8") as f:
        for line in f:
            if len(line.rstrip().split(" ")) <= 2: continue
            token, vector = line.rstrip().split(" ", 1)
            embedding_dict[token] = np.fromstring(vector, dtype=np.float, sep=" ")
    return embedding_dict


def make_embedding(embedding_file, emb_size, word2idx):

    embedding_dict = dict()
    embedding_dict["<unk>"] = np.array([0. for _ in range(emb_size)])
    _load_embedding(embedding_file, embedding_dict)
    logging.info("total embedding size is {} ".format(len(embedding_dict)))

    # emb_mat = [embedding_dict[token] for token in vocab if token in embedding_dict]
    #
    # index = 0
    # for token in embedding_dict.keys():
    #     if token in vocab:
    #         self.word2idx.update({token: index})
    #         index += 1

    count = 0
    emb_mat = [0 for _ in range(len(word2idx))]
    for token, v in word2idx.items():
        if token in embedding_dict.keys():
            emb_mat[v] = embedding_dict[token]
            emb_mat.append(embedding_dict[token])
            count += 1
        else:
            emb_mat[v] = embedding_dict["<unk>"]
    logging.info(
        "{} / {} tokens have corresponding in embedding vector".format(len(word2idx) - count, len(word2idx)))
    logging.info("total word vocabulary size is {} ".format(len(word2idx)))

    return emb_mat

import sys

def eval_data(model, data_loader):
    model.eval()

    start_time = time.time()
    entity_hit_num = 0.0
    entity_pred_num = 0.0
    entity_gold_num = 0.0

    spo_hit_num = 0.0
    spo_pred_num = 0.0
    spo_gold_num = 0.0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
            batch_char_ids, batch_word_ids, batch_sentence_len, batch_gold_answer = batch
            subject_preds, po_pred, data_idx = model(batch_char_ids, batch_word_ids, is_train=False, sentence_lens=batch_sentence_len)

            for gold_answer in batch_gold_answer:
                spo_gold_num += len(gold_answer)

            if len(data_idx) == 0:
                continue

            # print(batch_gold_answer)
            print(data_idx)
            print(subject_preds.shape, po_preds.shape)
            for i, (subject, po_pred) in enumerate(zip(subject_preds.data.cpu().numpy(), po_preds.data.cpu().numpy())):
                gold_answer = batch_gold_answer[i]

                sentence_len = batch_sentence_len[i]

                start = np.where(po_pred[:, :, 0] > 0.5)
                end = np.where(po_pred[:, :, 1] > 0.4)

                spoes = []
                for _start, predicate1 in zip(*start):
                    if _start >= sentence_len:
                        continue
                    for _end, predicate2 in zip(*end):
                        if _start <= _end < sentence_len and predicate1 == predicate2:
                            spoes.append((subject, predicate1, (_start, _end)))
                            break
                po_predict = []
                for s, p, o in spoes:

                    po_predict.append((s[0], s[1], o[0], o[1], p))

                    if (s[0], s[1], o[0], o[1], p) in gold_answer:
                        spo_hit_num += 1

                    spo_pred_num += 1
                # answer_dict[i][0].append(subject[0], subject[1])
                # answer_dict[i][1].extend(po_predict)
        print('============================================')
        print("em: {},\tpre&gold: {}\t{} ".format(spo_hit_num, spo_pred_num, spo_gold_num))
        eval_metrix_res = eval_metrix(spo_hit_num, spo_gold_num, spo_pred_num)
        print("f1: {}, \tPrecision: {},\tRecall: {} ".format(eval_metrix_res["f1_value"] * 100,
                                                             eval_metrix_res["precision"] * 100,
                                                             eval_metrix_res["recall"] * 100))
        # return {'f1': f1, "recall": recall, "precision": precision}

        logging.info("eval cost {}".format(time.time()-start_time))

            # self.forward(batch, chosen, eval=True, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--rel_num", type=int, default=len(data_loader.relation2id), required=False)
    parser.add_argument("--char_size", type=int, default=len(data_loader.char2id), required=False)
    parser.add_argument("--char_embed_size", type=int, default=128, required=False)
    parser.add_argument("--word_embed_size", type=int, default=300, required=False)
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--num_rnn_layers", type=int, default=2, required=False)
    parser.add_argument("--dropout", type=float, default=0.5, required=False)
    parser.add_argument("--encoder_head", type=int, default=4, required=False)
    parser.add_argument("--rnn_encoder", type=str, default="lstm", required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--embedding_file', type=str,
                        default='D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char', required=False)

    config = parser.parse_args()

    dataset = Duie2Dataset(data_loader.documents, data_loader.char2id, data_loader.word2id, config.rel_num)
    dev_dataset = Duie2Dataset(data_loader.dev_documents, data_loader.char2id, data_loader.word2id, config.rel_num, is_train=False)

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    dev_data_loader = dev_dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    word_embed = make_embedding(config.embedding_file, config.word_embed_size, data_loader.word2id)
    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
    model = EtlSpan(config, word_embed)
    parameters_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters_trainable, lr=config.learning_rate)

    step_gap = 20
    global_loss = 0.0
    for epoch in range(config.epoch):
        for step, batch in enumerate(train_data_loader):
            model.train()
            batch_char_ids, batch_word_ids, batch_token_type_ids, batch_subject_ids, batch_subject_labels, batch_object_labels, batch_sentence_len = batch

            sub_preds, po_preds, mask = model(batch_char_ids, batch_word_ids, batch_subject_ids, sentence_lens=batch_sentence_len)

            subject_loss = loss_fct(sub_preds, batch_subject_labels)
            subject_loss = subject_loss.mean(2)
            subject_loss = torch.sum(subject_loss * mask.float()) / torch.sum(mask.float())

            # print(po_preds.shape, batch_subject_labels.shape)
            po_loss = loss_fct(po_preds, batch_object_labels)
            po_loss = torch.sum(po_loss.mean(3), 2)
            po_loss = torch.sum(po_loss * mask.float()) / torch.sum(mask.float())

            loss = subject_loss + po_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if step % step_gap == 0:
                global_loss += loss
                current_loss = global_loss / step_gap
                print(
                    u"step {} / {} of epoch {}, train/loss: {}".format(step, len(train_data_loader),
                                                                       epoch, current_loss))
                global_loss = 0.0

            if step and step % 100:
                eval_data(model, dev_data_loader)
            # if step and step % 100 == 0:




    # train_data_loader = DataLoader(dataset, config.batch_size, shuffle=True, collate_fn=)



