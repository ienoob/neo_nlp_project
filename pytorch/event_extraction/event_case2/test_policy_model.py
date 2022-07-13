#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import csv
import pandas as pd
from pytorch.event_extraction.event_case2.data_gen import Dataset, extract_info
from pytorch.event_extraction.event_case2.policy_information_extractor_v3 import Document
data_path = "D:\data\\tianjin_dataset\政策信息抽取\\20211129"
data_path = "D:\data\\tianjin_dataset\policy_20211129.csv"

dataset = Dataset(data_path, "list-json")
#
# i = -1
# limit = 2
# doc = Document()
# for data in dataset:
#     i += 1
#     if i < limit:
#         continue
#
#
#     res = extract_info(doc, data)
#     # print(res)
#     print(json.dumps(res, indent=4, ensure_ascii=False))
#     print("=============================== {}".format(i))
#
#     break
df = pd.read_csv(data_path,engine='python',quoting=csv.QUOTE_NONE)
