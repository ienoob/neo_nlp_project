#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    多任务模型，分词+词性+句法依存
"""
import torch
import numpy as np
from collections import defaultdict
from torch import nn
from transformers import ElectraModel
from pytorch.syntactic_parsing.parser_layer import NonLinear, Biaffine


class Tarjan:
  """
    Computes Tarjan's algorithm for finding strongly connected components (cycles) of a graph

    Attributes:
      edges: dictionary of edges such that edges[dep] = head
      vertices: set of dependents
      SCCs: list of sets of strongly connected components. Non-singleton sets are cycles.
  """

  # =============================================================
  def __init__(self, prediction, tokens):
    """
      Inputs:
        prediction: a predicted dependency tree where
          prediction[dep_idx] = head_idx
        tokens: the tokens we care about (i.e. exclude _GO, _EOS, and _PAD)
    """

    self._edges = defaultdict(set)
    self._vertices = set((0,))
    for dep, head in enumerate(prediction[tokens]):
      self._vertices.add(dep + 1)
      self._edges[head].add(dep + 1)
    self._indices = {}
    self._lowlinks = {}
    self._onstack = defaultdict(lambda: False)
    self._SCCs = []

    index = 0
    stack = []
    for v in self.vertices:
      if v not in self.indices:
        self.strongconnect(v, index, stack)

  # =============================================================
  def strongconnect(self, v, index, stack):
    """"""

    self._indices[v] = index
    self._lowlinks[v] = index
    index += 1
    stack.append(v)
    self._onstack[v] = True
    for w in self.edges[v]:
      if w not in self.indices:
        self.strongconnect(w, index, stack)
        self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
      elif self._onstack[w]:
        self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])

    if self._lowlinks[v] == self._indices[v]:
      self._SCCs.append(set())
      while stack[-1] != v:
        w = stack.pop()
        self._onstack[w] = False
        self._SCCs[-1].add(w)
      w = stack.pop()
      self._onstack[w] = False
      self._SCCs[-1].add(w)
    return

  # ======================
  @property
  def edges(self):
    return self._edges

  @property
  def vertices(self):
    return self._vertices

  @property
  def indices(self):
    return self._indices

  @property
  def SCCs(self):
    return self._SCCs

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def softmax2d(x, length1, length2):
    y = np.zeros((length1, length2))
    for i in range(length1):
        for j in range(length2):
            y[i,j] = x[i, j]
    y -= np.max(y, axis=1, keepdims=True)
    y = np.exp(y)
    return y / np.sum(y, axis=1, keepdims=True)

def arc_argmax(parse_probs, length, ensure_tree = True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        I = np.eye(length)
        # block loops and pad heads
        parse_probs = parse_probs * (1-I)
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length)
        roots = np.where(parse_preds[tokens] == 0)[0]+1
        # ensure at least one root
        if len(roots) < 1:
            # The current root probabilities
            root_probs = parse_probs[tokens,0]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = 0
            # ensure at most one root
        elif len(roots) > 1:
            # The probabilities of the current heads
            root_probs = parse_probs[roots,0]
            # Set the probability of depending on the root zero
            parse_probs[roots,0] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:,tokens], axis=1)+1
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:,tokens], axis=1)+1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds

def rel_argmax(rel_probs, length, ROOT, ensure_tree = True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        rel_probs[0] = 0
        root = ROOT
        tokens = np.arange(1, length)
        rel_preds = np.argmax(rel_probs, axis=1)
        roots = np.where(rel_preds[tokens] == root)[0]+1
        if len(roots) < 1:
            rel_preds[1+np.argmax(rel_probs[tokens,root])] = root
        elif len(roots) > 1:
            root_probs = rel_probs[roots, root]
            rel_probs[roots, root] = 0
            new_rel_preds = np.argmax(rel_probs[roots], axis=1)
            new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
            new_root = roots[np.argmin(new_rel_probs)]
            rel_preds[roots] = new_rel_preds
            rel_preds[new_root] = root
        return rel_preds
    else:
        rel_probs[0] = 0
        rel_preds = np.argmax(rel_probs, axis=1)
        return rel_preds

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)


class BiaffineParser(nn.Module):

    def __init__(self, config):
        super(BiaffineParser, self).__init__()
        self.root = 2
        self.config = config

        self.mlp_arc_dep = NonLinear(
            input_size=config.hidden_size,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size=config.hidden_size,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, config.rel_size, bias=(True, True))

    def forward(self, input_hidden, masks):


        x_all_dep = self.mlp_arc_dep(input_hidden)
        x_all_head = self.mlp_arc_head(input_hidden)

        x_all_dep_splits = torch.split(x_all_dep, split_size_or_sections=100, dim=2)
        x_all_head_splits = torch.split(x_all_head, split_size_or_sections=100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond

    def parse(self, input_hidden, lengths, masks):
        arc_logits, rel_logits = self.forward(input_hidden, masks)
        ROOT = self.root
        arcs_batch, rels_batch = [], []
        arc_logits = arc_logits.data.cpu().numpy()
        rel_logits = rel_logits.data.cpu().numpy()

        for arc_logit, rel_logit, length in zip(arc_logits, rel_logits, lengths):
            arc_probs = softmax2d(arc_logit, length, length)
            arc_pred = arc_argmax(arc_probs, length)

            rel_probs = rel_logit[np.arange(len(arc_pred)), arc_pred]
            rel_pred = rel_argmax(rel_probs, length, ROOT)

            arcs_batch.append(arc_pred)
            rels_batch.append(rel_pred)

        return arcs_batch, rels_batch

    def compute_accuracy(self, true_arcs, true_rels, arc_logits, rel_logits):
        b, l1, l2 = arc_logits.size()
        pred_arcs = arc_logits.data.max(2)[1].cpu()
        index_true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        arc_correct = pred_arcs.eq(true_arcs).cpu().sum()


        size = rel_logits.size()
        output_logits = _model_var(self, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][arcs[i]])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        pred_rels = output_logits.data.max(2)[1].cpu()
        true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        label_correct = pred_rels.eq(true_rels).cpu().sum()

        total_arcs = b * l1 - np.sum(true_arcs.cpu().numpy() == -1)

        return arc_correct, label_correct, total_arcs


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


def seg_sequence(input_seq):
    seg_res = []
    start = 1
    for i, si in enumerate(input_seq):
        if i == 0:
            continue
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


class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super(MultiTaskModel, self).__init__()
        self.config = config
        self.electra_model = ElectraModel.from_pretrained(config.electra_model_name)
        self.seg = nn.Linear(config.hidden_size, config.seg_size)
        self.softmax = nn.Softmax(2)
        self.pos = nn.Linear(config.hidden_size, config.pos_size)
        self.dep = BiaffineParser(config)

    def forward(self, input_ids, word_idx, masks=None, pre_train_mask=None):
        seg_id = torch.zeros(pre_train_mask.size()).long()
        encoder = self.electra_model(input_ids, pre_train_mask, seg_id)
        encoder = encoder[0]
        # print(encoder.shape)
        seg_logits = self.seg(encoder)

        seg_logits = self.softmax(seg_logits)
        # print(word_idx.shape)
        word_idx = word_idx.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        word_input = torch.gather(encoder, dim=1, index=word_idx)

        pos_logits = self.pos(word_input)

        arc_logit, rel_logit_cond = self.dep.forward(word_input, masks)

        return seg_logits, pos_logits, arc_logit, rel_logit_cond

    def predict(self, input_ids, char_masks):
        encoder = self.electra_model(input_ids)
        encoder = encoder[0]
        # print(encoder.shape)
        seg_logits = self.seg(encoder)

        seg_logits = self.softmax(seg_logits)

        seg_predicts = torch.argmax(seg_logits, dim=-1)
        word_idxs = []
        seq_segs = []
        masks = []
        lengths = []
        length = 0
        for i, seg_predict in enumerate(seg_predicts):
            char_length = int(char_masks[i].sum())

            seg_predict = seg_predict.squeeze(-1)
            seq_seg = seg_sequence(seg_predict[:char_length])

            length = max(length, len(seq_seg))
            mask = [1]*len(seq_seg)
            word_idx = [x[0] for x in seq_seg]
            word_idxs.append(np.array(word_idx))
            lengths.append(len(word_idx))
            masks.append(np.array(mask))
            seq_segs.append(seq_seg)

            if len(seq_seg) == 0:
                print(seg_predict[:char_length])

        word_idxs = pad_sequence(word_idxs, length, padding=0, dtype=np.int64)
        masks = pad_sequence(masks, length, padding=0,  dtype=np.int64)
        word_idxs = word_idxs.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        word_input = torch.gather(encoder, dim=1, index=word_idxs)
        pos_logits = self.pos(word_input)
        pos_logits = self.softmax(pos_logits)
        pos_predicts = torch.argmax(pos_logits, dim=-1)
        #
        arcs_batch, rels_batch = self.dep.parse(word_input, lengths, masks)

        return seq_segs, pos_predicts, arcs_batch, rels_batch
        #
        # # print(word_idxs)
        #
        #
        #
        # if word_input.shape[1] == 0:
        #     batch_num = word_input.shape[0]
        #     return word_idxs, torch.tensor([[] for _ in range(batch_num)]), torch.tensor([[] for _ in range(batch_num)]), torch.tensor([[] for _ in range(batch_num)])
        #
        #
        # return word_idxs,


class MultiTaskModelV2(nn.Module):
    def __init__(self, config):
        super(MultiTaskModelV2, self).__init__()
        self.config = config
        self.electra_model = ElectraModel.from_pretrained(config.bert_model_name)
        self.seg = nn.Linear(config.hidden_size, config.seg_size)

    def forward(self, input_ids, attention_masks, token_type_ids):
        encoder = self.electra_model(input_ids, attention_masks, token_type_ids)
        encoder = encoder[0][:,1:-1,:]

        seg_logits = self.seg(encoder)

        return seg_logits



if __name__ == "__main__":

    pass







