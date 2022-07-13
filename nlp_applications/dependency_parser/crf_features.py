#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2020/12/28 20:12
    @Author  : jack.li
    @Site    : 
    @File    : crf_features.py

"""

import re
from collections import defaultdict
from crf_model_v1 import FeatureSet

alphas = re.compile('^[a-zA-Z]+$')


def fit_dataset(filename):
    labels = set()
    obsrvs = set()
    word_sets = defaultdict(set)

    sents_words = []
    sents_labels = []

    for line in open(filename, 'r', encoding="utf-8"):
        sent_words = []
        sent_labels = []
        try:
            for token in line.strip().split():
                word, label = token.rsplit('/', 2)
                orig_word = word
                word = word.lower()
                labels.add(label)
                obsrvs.add(word)
                word_sets[label].add(word)
                sent_words.append(orig_word)
                sent_labels.append(label)
            sents_words.append(sent_words)
            sents_labels.append(sent_labels)
        except Exception:
            print(line)
    return (labels, obsrvs, word_sets, sents_words, sents_labels)


class Membership(FeatureSet):
    def __init__(self, label, word_set):
        self.label = label
        self.word_set = word_set

    def __call__(self, yp, y, x_v, i):
        if i < len(x_v) and y == self.label and (x_v[i].lower() in self.word_set):
            return 1
        else:
            return 0


class FileMembership(Membership):
    @classmethod
    def functions(cls, lbls, *filenames):
        sets = [
            set([line.strip().lower() for line in open(filename, 'r')])
            for filename in filenames
        ]
        return super(FileMembership, cls).functions(lbls, *sets)


class MatchRegex(FeatureSet):
    def __init__(self, label, regex):
        self.label = label
        self.regex = re.compile(regex)

    def __call__(self, yp, y, x_v, i):
        if i < len(x_v) and y == self.label and self.regex.match(x_v[i]):
            return 1
        else:
            return 0


if __name__ == "__main__":
    val = Membership.functions(['HI', 'HO'], set(['hi']), set(['ho'])) + MatchRegex.functions(['HI', 'HO'], '\w+',
                                                                                              '\d+')
    print(val[0]('HO', 'HO', ['hi'], 0))
    print(val[0]('HO', 'HI', ['hi'], 0))
