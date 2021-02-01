#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/24 22:36
    @Author  : jack.li
    @Site    : 
    @File    : word_seqment.py

"""
from nlp_applications.word_segment.load_words import load_dic

word_dict = load_dic()


class TrieTree(object):

    def __init__(self, val=None):
        self.next = dict()
        self.val = val
        self.fail_index = None
        self.is_leaf = False
        self.word_ind = -1


class AC(object):

    def __init__(self):
        self.root = TrieTree()
        self.index = 0

    def build_word(self, word):

        rt = self.root

        for w in word:
            if w not in rt.next:
                rt.next[w] = TrieTree(w)
            rt = rt.next[w]

        rt.is_leaf = True
        rt.word_ind = self.index
        self.index += 1

    def build_tree(self, words):
        for word in words:
            self.build_word(word)
        self.build_fail_index()

    def build_fail_index(self):
        rt = self.root
        rt.fail_index = rt
        popv = []
        for _, v in rt.next.items():
            v.fail_index = self.root
            popv.append(v)

        while popv:
            p = popv.pop(0)
            p_fail_node = p.fail_index
            for k, v in p.next.items():
                if k in p_fail_node.next:
                    v.fail_index = p_fail_node.next[k]
                else:
                    v.fail_index = self.root
                popv.append(v)

    def search(self, input_str):
        rt = self.root
        words = []
        for i, s in enumerate(input_str):

            if s in rt.next:
                rt = rt.next[s]
            else:
                while rt and rt != self.root:
                    if s in rt.next:
                        rt = rt.next[s]
                        break
                    rt = rt.fail_index
                if rt is None:
                    rt = self.root
            nt = rt
            while nt and nt != self.root:
                if nt.is_leaf:
                    words.append((nt.word_ind, i))
                nt = nt.fail_index
        return words


ac = AC()
word_list = list(word_dict.keys())

word_str = "想通过 Kaggle 磨练数据科学技能？先听听 Kaggle Grandmaster 分享了哪些成功经验"

ac.build_tree(word_list)

word_find = ac.search(word_str)
word_count = dict()
for ind, loc in word_find:
    start_id = loc-len(word_list[ind])+1
    word_count.setdefault(start_id, [0, 0])
    i, x = word_count[start_id]
    if len(word_list[ind]) > x:
        word_count[start_id] = [ind, len(word_list[ind])]

print(word_count)
word_seg = []
i = 0
last = 0
while i < len(word_str):
    if i in word_count:
        word_seg.append(word_str[last:i])
        ind, x = word_count[i]
        word_seg.append(word_list[ind])

        i += x
        last = i
    i += 1

if last < len(word_str):
    word_seg.append(word_str[last:])
print(word_seg)

