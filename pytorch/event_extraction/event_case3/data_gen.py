#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
from nlp_applications.data_loader import load_json_line_data


train_path = "D:\data\篇章级事件抽取\\duee_fin_train.json\\duee_fin_train.json"
dev_path = "D:\data\篇章级事件抽取\\duee_fin_dev.json\\duee_fin_dev.json"
new_data_path = "D:\data\\tj_event\\trz_events.csv"

train_data = load_json_line_data(train_path)
dev_data = load_json_line_data(dev_path)
role_set = set()
trigger_set = set()

role2id = dict()

role_max_statis = dict()
def f_data(input_data):
    documents = []
    # sub_trigger_set = set()
    for i, sub_train_data in enumerate(input_data):
        text = sub_train_data["text"]
        title = sub_train_data["title"]
        doc_id = sub_train_data["id"]

        event_list = []
        for sub_event in sub_train_data.get("event_list", []):
            if sub_event["event_type"] == "中标":
                event_list.append(sub_event)
                trigger_set.add(sub_event["trigger"])
        #
        if doc_id == "449d3a2a83db545007acd62940191db0":
            for event in event_list:
                if event["arguments"][2]["role"] == "中标金额" and event["arguments"][5]["role"] == "中标金额":
                    event["arguments"] = event["arguments"][:-1]
        elif doc_id == "58bb7dd72a013eda5fe0b93325db0764":
            for event in event_list:
                if event["arguments"][0]["role"] == "中标金额" and event["arguments"][4]["role"] == "中标金额":
                    event["arguments"] = event["arguments"][1:]
        elif doc_id == "bee0efc875a700f4cc19cadabf62c750":
            for event in event_list:
                if event["arguments"][2]["role"] == "中标金额" and event["arguments"][3]["role"] == "中标金额":
                    event["arguments"][3]["role"] = "中标日期"
        elif doc_id == "7fe50e75d191801168131bb55ad1de9f":
            for event in event_list:
                if event["arguments"][-1]["role"] == "中标日期" and event["arguments"][-1]["argument"] == "日前":
                    event["arguments"][-1]["role"] = "披露日期"
        elif doc_id == "d43725c0578434234674ac657a7144d9":
            for event in event_list:
                if event["arguments"][0]["role"] == "中标公司" and event["arguments"][0]["argument"] == "浙江华海药业股份有限公":
                    event["arguments"][0]["argument"] = "浙江华海药业股份有限公司"
        elif doc_id == "4c5ade42d77dc9c175e4ca0abf4a8a0f":
            for event in event_list:
                if event["arguments"][2]["role"] == "招标方" and event["arguments"][3]["role"] == "招标方" and event["arguments"][4]["role"] == "招标方":
                    event["arguments"] = event["arguments"][:-3]

        if len(event_list):
            for event in event_list:
                sub_role_statis = dict()
                for arg in event["arguments"]:
                    if arg["role"] not in role2id:
                        role2id[arg["role"]] = len(role2id)

                    sub_role_statis.setdefault(arg["role"], 0)
                    sub_role_statis[arg["role"]] += 1

                for k, v in sub_role_statis.items():
                    role_max_statis[k] = max(role_max_statis.get(k, 0), v)
                if sub_role_statis.get("招标方", 0) == 3:
                    print(text)
                    print(event)
                    print(doc_id)


            documents.append({"id": doc_id, "text": text, "title": title, "event": event_list})
        else:
            continue

    return documents


train_documents = f_data(train_data)
dev_documents = f_data(dev_data)

print(role2id)

print(trigger_set)
print(role_max_statis)
documents = train_documents + dev_documents

with open("bidding.json", "w") as f:
    f.write(json.dumps(documents))
# print(role_set)
