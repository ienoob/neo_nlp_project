#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import argparse
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from pytorch.syntactic_parsing.parser import BiaffineParser


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


def data_iter(batch_num, input_data_path=train_data_path):
    batch_data = []
    for sentence in generator_sentence(input_data_path):
        batch_data.append(sentence)
        if len(batch_data) == batch_num:
            yield batch_data
            batch_data = []


word2id = {
    "<pad>": 0,
    "<unk>": 1,
    "<root>": 2
}

tag2id = {
    "<pad>": 0,
    "<unk>": 1
}

rel2id = {
    "<pad>": 0,
    "<unk>": 1
}

extword2id = {
    "<pad>": 0,
    "<unk>": 1,
    "<root>": 2
}

for sentence in generator_sentence():
    for dep in sentence:
        word = dep[1]
        tag = dep[3]
        rel = dep[7]

        if word not in word2id:
            word2id[word] = len(word2id)
            extword2id[word] = len(extword2id)
        if tag not in tag2id:
            tag2id[tag] = len(tag2id)
        if rel not in rel2id:
            rel2id[rel] = len(rel2id)

print(len(word2id))
print(len(tag2id))
print(len(rel2id))

id2rel = {v: k for k, v in rel2id.items()}


def sentences_numberize(sentences):
    for sentence in sentences:
        yield sentence2id(sentence)


def sentence2id(sentence):
    result = []
    for dep in sentence:
        wordid = word2id.get(dep[1], word2id["<unk>"])
        extwordid = extword2id.get(dep[1], word2id["<unk>"])
        tagid = tag2id[dep[3]]
        head = int(dep[6])
        relid = rel2id[dep[7]]
        result.append([wordid, extwordid, tagid, head, relid])

    return result


def batch_data_variable(batch, vocab):
    length = len(batch[0])
    batch_size = len(batch)
    for b in range(1, batch_size):
        if len(batch[b]) > length: length = len(batch[b])

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []

    b = 0
    for sentence in sentences_numberize(batch):
        index = 0
        length = len(sentence)
        lengths.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        for dep in sentence:
            words[b, index] = dep[0]
            extwords[b, index] = dep[1]
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            masks[b, index] = 1
            index += 1
        b += 1
        heads.append(head)
        rels.append(rel)

    return words, extwords, tags, heads, rels, lengths, masks

# from nlp_applications.utils import load_word_vector
#
# word_embed_path = "D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char"
# word_embed = load_word_vector(word_embed_path)
# print(word_embed.keys())

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

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


class Dependency:
    def __init__(self, id, form, tag, head, rel):
        self.id = id
        self.org_form = form
        self.form = form.lower()
        self.tag = tag
        self.head = head
        self.rel = rel

    def __str__(self):
        values = [str(self.id), self.org_form, "_", self.tag, "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)

    @property
    def pseudo(self):
        return self.id == 0 or self.form == '<eos>'

def batch_variable_depTree(trees, heads, rels, lengths, vocab):
    for tree, head, rel, length in zip(trees, heads, rels, lengths):
        sentence = []
        for idx in range(length):
            # print(tree[idx])
            sentence.append(Dependency(idx, tree[idx][1], tree[idx][3], head[idx], id2rel[rel[idx]]))
        yield sentence

def evalDepTree(gold, predict):
    PUNCT_TAGS = ['``', "''", ':', ',', '.', 'PU']
    ignore_tags = set(PUNCT_TAGS)
    start_g = 0
    if gold[0].id == 0: start_g = 1
    start_p = 0
    if predict[0].id == 0: start_p = 1

    glength = len(gold) - start_g
    plength = len(predict) - start_p

    if glength != plength:
        raise Exception('gold length does not match predict length.')

    arc_total, arc_correct, label_total, label_correct = 0, 0, 0, 0
    for idx in range(glength):
        if gold[start_g + idx].pseudo: continue
        if gold[start_g + idx].tag in ignore_tags: continue
        arc_total += 1
        label_total += 1
        if gold[start_g + idx].head == predict[start_p + idx].head:
            arc_correct += 1
            if gold[start_g + idx].rel == predict[start_p + idx].rel:
                label_correct += 1

    return arc_total, arc_correct, label_total, label_correct

def evaluate(model, config):
    start = time.time()
    model.eval()
    # output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(config.test_batch_size, input_data_path=dev_data_path):
        words, extwords, tags, heads, rels, lengths, masks = \
            batch_data_variable(onebatch, None)
        count = 0
        arcs_batch, rels_batch = model.parse(words, extwords, tags, lengths, masks)
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, None):
            # printDepTree(output, tree)
            preds = [Dependency(item[0], item[1], item[3], int(item[6]), item[7]) for item in onebatch[count]]
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, preds)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    # output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (0, during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


def load_pretrain_embbeding():
    pretrain_embed = np.zeros((len(extword2id), 100))
    model = Word2Vec.load('word2vec.model')
    for word in model.wv.vocab:
        word_id = extword2id[word]
        pretrain_embed[word_id] = model.wv[word]

    return pretrain_embed

class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()



def train():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vocab_size", type=int, default=len(word2id), required=False)
    parser.add_argument("--extvocab_size", type=int, default=len(extword2id), required=False)
    parser.add_argument("--tag_size", type=int, default=len(tag2id), required=False)
    parser.add_argument("--rel_size", type=int, default=len(rel2id), required=False)
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
    parser.add_argument("--test_batch_size", type=int, default=50, required=False)
    parser.add_argument("--train_batch_size", type=int, default=50, required=False)
    parser.add_argument("--learning_rate", type=float, default=2e-3, required=False)
    parser.add_argument("--beta_1", type=float, default=.9, required=False)
    parser.add_argument("--beta_2", type=float, default=.9, required=False)
    parser.add_argument("--decay", type=float, default=.75, required=False)
    parser.add_argument("--decay_steps", type=int, default=5000, required=False)
    parser.add_argument("--epsilon", type=float, default=1e-12, required=False)

    config = parser.parse_args()

    pretrain_embed = load_pretrain_embbeding()
    print(pretrain_embed.shape)
    model = BiaffineParser(config, pretrain_embed)

    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adamax(parameters)
    optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), config)

    global_step = 0
    for epoch in range(100):
        start_time = time.time()
        print('epoch: ' + str(epoch))
        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for batch_iter, one_batch in enumerate(data_iter(config.train_batch_size)):
            global_step += 1
            words, extwords, tags, heads, rels, lengths, masks = \
                batch_data_variable(one_batch, None)
            model.train()
            arc_logit, rel_logit_cond = model.forward(words, extwords, tags, masks)
            loss = compute_loss(model, heads, rels, arc_logit, rel_logit_cond, lengths)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()

            # print(loss_value)
            loss.backward()

            arc_correct, label_correct, total_arcs = model.compute_accuracy(heads, rels, arc_logit, rel_logit_cond)
            overall_arc_correct += arc_correct
            overall_label_correct += label_correct
            overall_total_arcs += total_arcs
            uas = overall_arc_correct * 100.0 / overall_total_arcs
            las = overall_label_correct * 100.0 / overall_total_arcs
            during_time = float(time.time() - start_time)
            print("Step:%d, ARC:%.2f, REL:%.2f, Iter:%d, batch:%d, length:%d,time:%.2f, loss:%.2f" \
                  % (global_step, uas, las, epoch, batch_iter, overall_total_arcs, during_time, loss_value))

            optimizer.step()
            model.zero_grad()

            if batch_iter % 100 == 0:
                arc_correct, rel_correct, arc_total, dev_uas, dev_las = \
                    evaluate(model, config)
                print("Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
                      (arc_correct, arc_total, dev_uas, rel_correct, arc_total, dev_las))


if __name__ == "__main__":
    train()
