#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/28 23:20
    @Author  : jack.li
    @Site    : 
    @File    : max_forward_seg.py

"""
"""
    正向最大分词
"""
from nlp_applications.word_segment.word_seqment import ac, word_list

def max_forward_seg(word_str):

    word_find = ac.search(word_str)
    word_count = dict()
    for ind, loc in word_find:
        start_id = loc-len(word_list[ind])+1
        word_count.setdefault(start_id, [0, 0])
        i, x = word_count[start_id]
        if len(word_list[ind]) > x:
            word_count[start_id] = [ind, len(word_list[ind])]

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

    return word_seg


print(max_forward_seg("想通过 Kaggle 磨练数据科学技能？先听听 Kaggle Grandmaster 分享了哪些成功经验"))