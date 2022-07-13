#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

def split_str(input_str: str, split_char=None):
    if split_char is None:
        split_char = {".", "?", "？", "!", "！", "\r", "\n", " "}
    not_add_char = {"\r", "\n", " "}
    start = 0

    for i, i_char in enumerate(input_str):
        if i_char not in split_char:
            continue
        if i > start:
            if i_char in not_add_char:
                sub_str = input_str[start:i]
            else:
                sub_str = input_str[start:i+1]
            sub_str = sub_str.strip()
            yield sub_str
        start = i+1
    if start < len(input_str):
        sub_str = input_str[start:]
        yield sub_str

