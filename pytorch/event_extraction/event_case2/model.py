#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json

with open("policy.json", "r") as f:
    data = f.read()

data_json = json.loads(data)

for data in data_json[1:]:
    print(data["content"])
    print(data["event"])
    break
