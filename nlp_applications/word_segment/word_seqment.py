#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/24 22:36
    @Author  : jack.li
    @Site    : 
    @File    : word_seqment.py

"""
from .load_words import load_dic

word_dict = load_dic()


class TrieTree(object):

    def __init__(self, val=None):
        self.next = dict()
        self.val = val
        self.is_leaf = False


class AC(object):

    def __init__(self):
        self.root = TrieTree()


    def build_word(self, word):

        rt = self.root

        for w in word:
            if w not in rt.next:
                rt.next[w] = TrieTree(w)
            rt = rt.next[w]

        rt.is_leaf = True

    def build_tree(self, words):
        for word in words:
            self.build_word(word)


    def build_fail_index(self):
        pass

    def search(self, input_str):

        pass


