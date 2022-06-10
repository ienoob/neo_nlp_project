#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
from pytorch.event_extraction.event_case1.bert_extractor import bert2extract_ner
with open("base_case.json", "r") as f:
    data = f.read()

data_json = json.loads(data)

for b_data in data_json[3:]:
    print(b_data["title"])
    print(b_data["text"])
    print(b_data["event"])
    print(b_data["predict"])
    print(b_data["id"])

    bert2extract_ner(b_data["text"])

    break
