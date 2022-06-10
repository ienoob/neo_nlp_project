#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import pandas as pd


data_path = "policy_data-20211228.csv"

df = pd.read_csv(data_path)

for idx, row in df.iterrows():
    print(row["text"])

    break
