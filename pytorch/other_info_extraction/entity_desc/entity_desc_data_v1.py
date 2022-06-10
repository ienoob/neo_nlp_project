#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import json
from nlp_tf2_implement.other_information_extract.entity_describe_data_p2 import generate_label_data

train_data = []
for item in generate_label_data():
    train_data.append(item)


with open("D:\data\self-data\entity_desc_v1.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_data))
