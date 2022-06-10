#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import json

with open("finance.json", "r") as f:
    datas = f.read()

with open("finance_add.json", "r") as f:
    data_add = f.read()


documents = json.loads(datas)
add_documents = json.loads(data_add)

documents += add_documents
icount = 0
icount_list = ["dfcc2a1868a6518892e54805679d269d"]
for doc in documents:
    iv = 0
    ivs = []

    if doc.get("title"):
        text = doc["title"] + "\n" + doc["text"]
    else:
        text = doc["text"]
    for event in doc["event"]:
        for arg in event["arguments"]:
            if arg["role"] == "领投方":
                items = re.findall(arg["argument"], text)
                if len(items) > 1:
                    iv = 1
                    ivs.append(arg["argument"])
    if iv:
        icount += 1
        # if icount == 104:
        if doc["id"] == "1b08c89e5092157a48dc0481785a73fc":
            print(doc["id"])
            # print(doc["event"])
            print(text)
            print(ivs)
print(icount)
