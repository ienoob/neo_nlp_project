#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import argparse
import numpy as np
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch.syntactic_parsing.multi_task_model import MultiTaskModel, MultiTaskModelV2
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3
from transformers import ElectraTokenizer
from functools import partial
from torch.utils.data import Dataset, DataLoader


electra_model_name = "hfl/chinese-electra-small-discriminator"
data_path = "D:\data\depency_parser\evsam05\依存分析训练数据\THU"
train_data_path = data_path + "\\" + "train.conll"
dev_data_path = data_path + "\\" + "dev.conll"


def generator_sentence(input_data_path=train_data_path):
    with open(input_data_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    cache = [[0, '<root>', '<root>', 'root', 'root', '_', '0', '核心成分']]
    for train_row in train_data.split("\n"):
        train_row = train_row.strip()
        if train_row == "":
            yield cache
            cache = [[0, '<root>', '<root>', 'root', 'root', '_', '0', '核心成分']]
        else:
            train_row_dep = train_row.split("\t")
            train_row_dep[0] = int(train_row_dep[0])
            assert len(train_row_dep) == 8
            cache.append(train_row_dep)
    if len(cache) > 1:
        yield cache

def generator_seg_sentence(input_data_path=train_data_path):
    with open(input_data_path, "r", encoding="utf-8") as f:
        train_data = f.read()
    data_list = []
    cache = []
    max_len = 0
    for train_row in train_data.split("\n"):
        train_row = train_row.strip()
        if train_row == "":
            seg_data = [x[1] for x in cache]
            raw_data = "".join(seg_data)
            max_len = max(max_len, len(raw_data))

            data_list.append({"seg_data": seg_data, "raw_data": raw_data})
            cache = []
        else:
            train_row_dep = train_row.split("\t")
            assert len(train_row_dep) == 8
            cache.append(train_row_dep)
    if len(cache) > 1:
        seg_data = [x[1] for x in cache]
        raw_data = "".join(seg_data)
        max_len = max(max_len, len(raw_data))

        data_list.append({"seg_data": seg_data, "raw_data": raw_data})
    return data_list, max_len

dataset, _ = generator_seg_sentence()
for data in dataset:
    if data["seg_data"]==1:
        print(data)



class MultiTask2Dataset(Dataset):

    def __init__(self, documents, bert_model_name, is_train=True, max_len=0):
        super(MultiTask2Dataset, self).__init__()
        self.documents, self.max_len = documents
        self.bert_model_name = bert_model_name
        self.is_train = is_train
        self.tokenizer = ElectraTokenizer.from_pretrained(bert_model_name)
        self.seg2id = {
            "O": 0,
            "B": 1,
            "I": 2
        }
        # self.max_len = max_len

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def _create_collate_fn(self, batch_first=False):

        def collate(documents):
            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_label_ids = []
            truncation = True
            batch_size = len(documents)
            max_len = 0
            if self.max_len:
                max_len = self.max_len+2
            else:
                max_len = max([len(doc["raw_data"]) for doc in documents])+2
            batch_input_ids = Variable(torch.LongTensor(batch_size, max_len).zero_(), requires_grad=False)
            batch_attention_mask = Variable(torch.LongTensor(batch_size, max_len).zero_(), requires_grad=False)
            batch_token_type_id = Variable(torch.LongTensor(batch_size, max_len).zero_(), requires_grad=False)
            batch_label_ids = Variable(torch.LongTensor(batch_size, max_len-2).zero_(), requires_grad=False)
            batch_seq_lens = []
            batch_seg_res = []

            for b, document in enumerate(documents):
                raw_data = document["raw_data"]
                seg_data = document["seg_data"]
                sentence_words = list(raw_data)
                seg_res = []
                batch_seq_lens.append(len(sentence_words))
                tokenized = self.tokenizer.encode_plus(
                    sentence_words,  truncation=truncation,
                    return_tensors="pt")
                sentence_id = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]
                token_type_ids = tokenized["token_type_ids"]
                for i, s in enumerate(sentence_id[0]):
                    batch_input_ids[b, i] = s
                    batch_attention_mask[b, i] = attention_mask[0][i]
                    batch_token_type_id[b, i] = token_type_ids[0][i]

                li = 0
                for word in seg_data:
                    batch_label_ids[b, li] = self.seg2id["B"]
                    la = li
                    li += 1
                    for _ in word[1:]:
                        batch_label_ids[b, li] = self.seg2id["I"]
                        li += 1
                    seg_res.append((la, li))
                batch_seg_res.append(seg_res)

            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_label_ids": batch_label_ids,
                "batch_seq_lens": batch_seq_lens,
                "batch_seg_res": batch_seg_res
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

class DataIter(object):

    def __init__(self):
        self.char2id = {
            "<pad>": 0,
            "<unk>": 1,
            "<root>": 2
        }
        self.tokenizer = ElectraTokenizer.from_pretrained(electra_model_name)
        self.seg2id = {
            "O": 0,
            "B": 1,
            "I": 2
        }
        self.pos2id = {
            "<pad>": 0
        }

        self.dep2id = {
            "<pad>": 0,
            "<unk>": 1
        }

        for sentence in generator_sentence():
            for dep in sentence:
                word = dep[1]
                pos = dep[3]
                rel = dep[7]

                if word != "<root>":
                    for char in word:
                        if char not in self.char2id:
                            self.char2id[char] = len(self.char2id)

                if pos not in self.pos2id:
                    self.pos2id[pos] = len(self.pos2id)

                if rel not in self.dep2id:
                    self.dep2id[rel] = len(self.dep2id)

    def single_process(self, input_sentence):
        # row_sentence = [seg[1] ]
        sentence_id = []
        seg_label = [self.seg2id["O"]]
        pos_label = []
        dep_label = []
        seg_index = []
        heads = []
        d = 0
        sentence_raw = []
        for seg in input_sentence:
            word = seg[1]
            pos = seg[3]
            rel = seg[7]
            head = int(seg[6])
            if word == "<root>":
                sentence_raw.append("<root>")
                seg_label.append(0)
            else:
                sentence_raw += list(word)
                for i, char in enumerate(word):

                    if i == 0:
                        seg_label.append(self.seg2id["B"])
                    else:
                        seg_label.append(self.seg2id["I"])

            seg_index.append(d+1)

            pos_label.append(self.pos2id[pos])
            dep_label.append(self.dep2id[rel])
            heads.append(head)
            if word == "<root>":
                d += 1
            else:
                d += len(word)
        seg_label.append(self.seg2id["O"])
        sentence_words = list(sentence_raw)
        truncation = True
        tokenized = self.tokenizer.encode_plus(
            sentence_words, padding=True, truncation=truncation,
            return_tensors=torch.tensor)
        sentence_id = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        token_type_ids = tokenized["token_type_ids"]
        # print(sentence_raw, len(sentence_raw))
        # print("sentence code length {}".format(len(sentence_id)))
        # print("seg code length {}".format(len(seg_label)))

        return {
            "sentence_id": sentence_id,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "seg_label": seg_label,
            "pos_label": pos_label,
            "dep_label": dep_label,
            "sentence": input_sentence,
            "char_length": len(sentence_id),
            "word_length": len(input_sentence),
            "seg_index": seg_index,
            "heads": heads
        }

    def batch_data_variable(self, batch):
        char_length_mx = 0
        word_length_mx = 0
        batch_size = len(batch)
        for b in range(batch_size):
            if batch[b]["char_length"] > char_length_mx:
                char_length_mx = batch[b]["char_length"]
            if batch[b]["word_length"] > word_length_mx:
                word_length_mx = batch[b]["word_length"]

        char_id = Variable(torch.LongTensor(batch_size, char_length_mx).zero_(), requires_grad=False)
        seg_label_list = Variable(torch.ones((batch_size, char_length_mx), dtype=torch.long)*-1, requires_grad=False)
        char_masks = Variable(torch.Tensor(batch_size, char_length_mx).zero_(), requires_grad=False)

        pos_label_list = Variable(torch.ones((batch_size, word_length_mx), dtype=torch.long)*-1, requires_grad=False)
        dep_label_list = Variable(torch.LongTensor(batch_size, word_length_mx).zero_(), requires_grad=False)
        heads_list = Variable(torch.LongTensor(batch_size, word_length_mx).zero_(), requires_grad=False)
        seg_index_list = Variable(torch.LongTensor(batch_size, word_length_mx).zero_(), requires_grad=False)
        word_masks = Variable(torch.Tensor(batch_size, word_length_mx).zero_(), requires_grad=False)

        heads = []
        rels = []
        lengths = []

        b = 0
        for b_data in batch:
            sentence_id = b_data["sentence_id"]
            seg_label = b_data["seg_label"]
            pos_label = b_data["pos_label"]
            dep_label = b_data["dep_label"]
            heads_i = b_data["heads"]
            seg_index = b_data["seg_index"]
            length = b_data["word_length"]
            head = np.zeros((length), dtype=np.int32)
            rel = np.zeros((length), dtype=np.int32)

            for i, s in enumerate(sentence_id):
                char_id[b, i] = s

                seg_label_list[b, i] = seg_label[i]
                char_masks[b, i] = 1

            for i, s in enumerate(pos_label):
                pos_label_list[b, i] = s
                dep_label_list[b, i] = dep_label[i]
                # heads_list[b, i] = heads[i]
                seg_index_list[b, i] = seg_index[i]
                word_masks[b, i] = 1
                head[i] = heads_i[i]
                rel[i] = dep_label[i]

            lengths.append(length)
            heads.append(head)
            rels.append(rel)

            b += 1

        return {
            "char_id": char_id,
            "seg_label_list": seg_label_list,
            "char_masks": char_masks,
            "pos_label_list": pos_label_list,
            "dep_label_list": rels,
            "heads_list": heads,
            "word_masks": word_masks,
            "seg_index": seg_index_list,
            "lengths": lengths
        }

    def train_iter(self, batch_num):
        batch_data = []
        for sentence in generator_sentence():
            batch_data.append(self.single_process(sentence))
            if len(batch_data) == batch_num:
                yield self.batch_data_variable(batch_data)
                batch_data = []
        if batch_data:
            yield self.batch_data_variable(batch_data)

    def dev_iter(self, batch_num):
        batch_data = []
        for sentence in generator_sentence(dev_data_path):
            batch_data.append(self.single_process(sentence))
            if len(batch_data) == batch_num:
                yield self.batch_data_variable(batch_data)
                batch_data = []
        if batch_data:
            yield self.batch_data_variable(batch_data)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def compute_loss(model, true_arcs, true_rels, arc_logits, rel_logits, lengths):
    b, l1, l2 = arc_logits.size()
    index_true_arcs = _model_var(
        model,
        pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
    true_arcs = _model_var(
        model,
        pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))

    masks = []
    for length in lengths:
        mask = torch.FloatTensor([0] * length + [-10000] * (l2 - length))
        mask = _model_var(model, mask)
        mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
        masks.append(mask.transpose(0, 1))
    length_mask = torch.stack(masks, 0)
    arc_logits = arc_logits + length_mask

    arc_loss = F.cross_entropy(
        arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
        ignore_index=-1)

    size = rel_logits.size()
    output_logits = _model_var(model, torch.zeros(size[0], size[1], size[3]))

    for batch_index, (logits, arcs) in enumerate(zip(rel_logits, index_true_arcs)):
        rel_probs = []
        for i in range(l1):
            rel_probs.append(logits[i][int(arcs[i])])
        rel_probs = torch.stack(rel_probs, dim=0)
        output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

    b, l1, d = output_logits.size()
    true_rels = _model_var(model, pad_sequence(true_rels, padding=-1, dtype=np.int64))

    rel_loss = F.cross_entropy(
        output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1)

    loss = arc_loss + rel_loss

    return loss

id2bio = {
    0: "O",
    1: "B",
    2: "I"
}


def seg_sequence(input_seq):
    seg_res = []
    start = 0
    for i, si in enumerate(input_seq):
        if si == 0:
            seg_res.append((i, i+1))
            start = i+1
        elif si == 1:
            if start < i:
                seg_res.append((start, i))
            start = i
    if start != len(input_seq):
        seg_res.append((start, len(input_seq)))
    return seg_res

import time
def evaluation(model, input_data_iter, config):
    start = time.time()
    model.eval()
    seg_hit = 0.0
    seg_real = 0.0
    seg_pred = 0.0

    pos_hit = 0.0
    pos_real = 0.0
    pos_pred = 0.0

    dep_hit_with_label = 0.0
    dep_hit_without_label = 0.0
    dep_real = 0.0
    dep_pred = 0.0

    for batch_data in input_data_iter.dev_iter(config.dev_batch_size):
        seg_predicts, pos_predicts, arcs_batch, rels_batch = model.predict(batch_data["char_id"], batch_data["char_masks"])

        seg_labels = batch_data["seg_label_list"]
        pos_labels = batch_data["pos_label_list"]
        rels = batch_data["dep_label_list"]
        heads = batch_data["heads_list"]

        # print(len(word_idx))
        # print(batch_data["char_masks"].shape)

        for b, seg_predict in enumerate(seg_predicts):
            length = int(torch.sum(batch_data["char_masks"][b]).data.numpy())
            # print(length)
            seg_label = seg_labels[b].data.numpy()
            seg_entity = seg_sequence(seg_label[:length])
            seg_real += len(seg_entity)
            pos_real += len(seg_entity)
            dep_real += len(seg_entity)
            #
            # seg_predict = seg_predict.data.numpy()
            # seg_predict_seq = [id2bio[s] for s in seg_predict[:length]]
            # seg_p_entity = seg_sequence(seg_predict_seq)
            #
            seg_pred += len(seg_predict)
            pos_pred += len(seg_predict)
            dep_pred += len(seg_predict)
            #
            for entity in seg_entity:
                if entity in seg_predict:
                    seg_hit += 1

            pos_label = pos_labels[b].data.numpy()
            # # print(len(seg_entity), len(pos_label))
            pos_predict = pos_predicts[b].data.numpy()
            # # print(len(seg_p_entity), len(pos_predict))
            #
            pos_entity = [(se, pos_label[i]) for i, se in enumerate(seg_entity) if i<len(pos_label)]
            pos_entity_pred = [(se, pos_predict[i]) for i, se in enumerate(seg_predict) if i<len(pos_predict)]
            #
            for p in pos_entity:
                if p in pos_entity_pred:
                    pos_hit += 1
            #
            rel = rels[b]
            head = heads[b]

            rel_pred = rels_batch[b]
            head_pred = arcs_batch[b]

            dep_entity = {se: (rel[i], head[i]) for i, se in enumerate(seg_entity) if i < len(rel)}
            dep_entity_pred = {se: (rel_pred[i], head_pred[i]) for i, se in enumerate(seg_predict) if i < len(rel_pred)}
            for k, v in dep_entity.items():
                if k not in dep_entity_pred:
                    continue
                v_head = seg_entity[v[1]]
                v_head_info = dep_entity[v_head]

                vv = dep_entity_pred[k]
                vv_head = seg_predict[vv[1]]
                vv_head_info = dep_entity_pred[vv_head]

                if v_head == vv_head:
                    dep_hit_without_label += 1
                    if v_head_info[0] == vv_head_info[0]:
                        dep_hit_with_label += 1

            # print(rels_batch[b])

    ev = eval_metrix_v3(seg_hit, seg_real, seg_pred)
    ev1 = eval_metrix_v3(pos_hit, pos_real, pos_pred)
    ev_dep_label = eval_metrix_v3(dep_hit_with_label, dep_real, dep_pred)
    ev_dep_u = eval_metrix_v3(dep_hit_without_label, dep_real, dep_pred)
    res = {
        "seg_hit":  seg_hit,
        "seg_real": seg_real,
        "seg_pred": seg_pred,
        "recall": ev["recall"],
        "precision": ev["precision"],
        "f1_value": ev["f1_value"],
        "pos_hit":  pos_hit,
        "pos_real": pos_real,
        "pos_pred": pos_pred,
        "pos_recall": ev1["recall"],
        "pos_precision": ev1["precision"],
        "pos_f1_value": ev1["f1_value"],
        "dep_hit_with_label": dep_hit_with_label,
        "dep_hit_without_label": dep_hit_without_label,
        "dep_real": dep_real,
        "dep_pred": dep_pred,
        "dep_uas": ev_dep_u["precision"],
        "dep_las": ev_dep_label["precision"]
    }

    print("eval cost {}".format(time.time()-start))

    return res


def train():

    data_iter = DataIter()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--char_size", type=int, default=len(data_iter.char2id), required=False)
    parser.add_argument("--seg_size", type=int, default=len(data_iter.seg2id), required=False)
    parser.add_argument("--pos_size", type=int, default=len(data_iter.pos2id), required=False)
    parser.add_argument("--rel_size", type=int, default=len(data_iter.dep2id), required=False)
    parser.add_argument("--word_dims", type=int, default=100, required=False)
    parser.add_argument("--tag_dims", type=int, default=100, required=False)
    parser.add_argument("--lstm_hiddens", type=int, default=400, required=False)
    parser.add_argument("--lstm_layers", type=int, default=2, required=False)
    parser.add_argument("--dropout_lstm_input", type=float, default=0.33, required=False)
    parser.add_argument("--dropout_lstm_hidden", type=float, default=0.33, required=False)
    parser.add_argument("--dropout_emb", type=float, default=0.33, required=False)
    parser.add_argument("--mlp_arc_size", type=int, default=500, required=False)
    parser.add_argument("--mlp_rel_size", type=int, default=100, required=False)
    parser.add_argument("--dropout_mlp", type=float, default=0.33, required=False)
    parser.add_argument("--update_every", type=int, default=4, required=False)
    parser.add_argument("--test_batch_size", type=int, default=10, required=False)
    parser.add_argument("--train_batch_size", type=int, default=10, required=False)
    parser.add_argument("--dev_batch_size", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=2e-4, required=False)
    parser.add_argument("--beta_1", type=float, default=.9, required=False)
    parser.add_argument("--beta_2", type=float, default=.9, required=False)
    parser.add_argument("--decay", type=float, default=.75, required=False)
    parser.add_argument("--decay_steps", type=int, default=5000, required=False)
    parser.add_argument("--epsilon", type=float, default=1e-12, required=False)
    parser.add_argument("--hidden_size", type=int, default=256, required=False)
    parser.add_argument("--electra_model_name", type=str, default=electra_model_name, required=False)

    config = parser.parse_args()

    model = MultiTaskModel(config)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    epoch = 100
    for e in range(epoch):

        for i, batch_data in enumerate(data_iter.train_iter(config.train_batch_size)):
            print("batch {} start".format(i))
            model.train()

            seg_logits, pos_logits, arc_logits, rel_logit_cond = model(batch_data["char_id"], batch_data["seg_index"],
                                                                       batch_data["word_masks"], batch_data["char_masks"])

            seg_labels = batch_data["seg_label_list"]
            pos_labels = batch_data["pos_label_list"]
            heads_list = batch_data["heads_list"]
            dep_label_list = batch_data["dep_label_list"]
            lengths = batch_data["lengths"]

            # masked_seg_logits = torch.masked_select(seg_logits, torch.ByteTensor(batch_data["char_masks"].byte()))
            # masked_seg_labels = torch.masked_select(seg_labels, torch.ByteTensor(batch_data["char_masks"].byte()))
            # print(cross_entropy3d(seg_logits, seg_labels, weight=None, size_average=True))
            loss_seg = criterion(seg_logits.view(-1, config.seg_size), seg_labels.view(-1))
            loss_pos = criterion(pos_logits.view(-1, config.pos_size), pos_labels.view(-1))
            loss_dep = compute_loss(model.dep, heads_list, dep_label_list, arc_logits, rel_logit_cond, lengths)

            loss = loss_seg + loss_pos + loss_dep
            print("epoch {0} batch {1} loss {2}".format(e, i, loss.data.numpy()))

            loss.backward()
            optimizer.step()
            model.zero_grad()

            if i and i % 10 == 0:
                print(evaluation(model, data_iter, config))


def evaluation_v2(model, dev_data_loader, config):
    model.eval()
    for batch in dev_data_loader:
        seg_logits = model(
            batch["batch_input_ids"],
            batch["batch_attention_mask"],
            batch["batch_token_type_id"])

        seg_preds = seg_logits.argmax(dim=-1)
        for b, seg_pred in enumerate(seg_preds):
            b_len = batch["batch_seq_lens"][b]
            seg_pred = seg_pred[:b_len]

            for i, s in enumerate(seg_pred):
                pass








def train_v2():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--bert_model_name", type=str, default="hfl/chinese-electra-small-discriminator", required=False)
    parser.add_argument("--word_dims", type=int, default=100, required=False)
    parser.add_argument("--tag_dims", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=256, required=False)
    parser.add_argument('--seg_size', type=int, default=3, required=False)
    parser.add_argument('--epoch', type=int, default=10, required=False)
    parser.add_argument('--device', type=str, default="cpu", required=False)
    config = parser.parse_args()

    dataset = MultiTask2Dataset(generator_seg_sentence(), config.bert_model_name)

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    model = MultiTaskModelV2(config)
    cross_en = nn.CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=config.learning_rate,
    #                      warmup=config.warmup_proportion,
    #                      t_total=int(len(dataset)//config.batch_size+1)*config.epoch)
    optimizer = torch.optim.Adamax(optimizer_grouped_parameters, lr=5e-5)

    for epoch in range(config.epoch):
        for step, batch in enumerate(train_data_loader):
            start = time.time()
            model.train()

            seg_logits = model(
                batch["batch_input_ids"],
                batch["batch_attention_mask"],
                batch["batch_token_type_id"])
            # print(seg_logits.shape)
            # print(batch["batch_label_ids"].shape)
            loss = cross_en(seg_logits.view(-1, config.seg_size), batch["batch_label_ids"].view(-1))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            break
        break


if __name__ == "__main__":
    train_v2()
