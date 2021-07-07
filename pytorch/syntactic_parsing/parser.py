#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable
from pytorch.syntactic_parsing.parser_layer import MyLSTM, NonLinear, Biaffine


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


def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class BiaffineParser(nn.Module):

    def __init__(self, config):
        super(BiaffineParser, self).__init__()
        self.root = 2
        self.config = config
        self.word_embed = nn.Embedding(config.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(config.extvocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(config.tag_size, config.tag_dims, padding_idx=0)

        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.mlp_arc_dep = NonLinear(
            input_size=2 * config.lstm_hiddens,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size=2 * config.lstm_hiddens,
            hidden_size=config.mlp_arc_size + config.mlp_rel_size,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, 1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, config.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks):
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

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

    def parse(self, words, extwords, tags, lengths, masks):
        if words is not None:
            arc_logits, rel_logits = self.forward(words, extwords, tags, masks)
        else:
            raise Exception()
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
