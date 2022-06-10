#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import json
import hashlib
# from pytorch.event_extraction.event_case1.test_finance_model import extractor



import pandas as pd


data_path = "D:\data\\tianjin_dataset\\tj_event\data\\"

for file in os.listdir(data_path):
    data_file = data_path + file
    try:
        with open(data_file, "r") as f:
            data = f.read()
    except UnicodeDecodeError as e:
        with open(data_file, "r", encoding="utf-8") as f:
            data = f.read()

    data_dict = json.loads(data)
    content = data_dict["content"]
    file_md5_indx = hashlib.md5(content.encode()).hexdigest()
    # print(data_dict["id"])
    if file_md5_indx == "1e44c6e9fdb1c01a614e47cb78b38f9e":
        print(file)
