#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import pandas as pd
from pytorch.event_extraction.event_case1.bert_extractor import bert2extract_ner



data_path = "D:\data\\tianjin_dataset\\tj_event\\trz_events.csv"

data = pd.read_csv(data_path, encoding="gbk")
correct_list = [436, 441, 454, 471, 476, 483, 492, 493]
# indx = 421
for indx in range(436, 2500):
    content = data["content"][indx]
    if isinstance(content, float):
        continue

    event_list = bert2extract_ner(content)
    print(indx)
    print(content)
    for event in event_list:
        print(event)
    break


