#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import json
from pytorch.other_info_extraction.policy_information_extractor_v3 import Document


class Dataset(object):
    def __init__(self, path, way="list"):
        self.path = path
        self.way = way

    def __iter__(self):
        if self.way == "list":
            for file in os.listdir(self.path):
                file_path = self.path + "\\" + file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
                yield data
        elif self.way == "jsonline":
            with open(self.path, "r", encoding="utf-8") as f:
                data = f.read()
            for dt in data.split("\n"):
                if dt.strip() == "":
                    continue
                # print(dt)
                dt_content = json.loads(dt)
                yield dt_content["content"]


if __name__ == "__main__":
    demo_path = "Z:\政策抽取\政策原文解析.txt"
    demo_path2 = "zc_json.jsonline"
    demo_path3 = "Z:\政策抽取\政策文件.jsonline"
    demo_path4 = "D:\data\政策信息抽取\\text"
    path = "Z:\政策抽取\政策信息抽取\chace_files\chace_files\\3产业基金"
    dataset = Dataset(demo_path4, "list")

    i = 0
    limit = 1
    doc = Document()
    for data in dataset:
        i += 1
        if i < limit:
            continue
        # print(data)
        doc.parse_content(data)
        doc.display_document()
        # print(doc.content)
        res = doc.parse_half_struction()
        print(json.dumps(res, ensure_ascii=False, indent=4))

        v = doc.parse_precision(res["conditions"])
        print(v["zhibiao"])
        print(v["for_train"])
        # doc.parse_contact_info()

        for k_project in res["project_infos"]:
            print("项目名称", k_project["project_name"])
            v = doc.parse_precision(k_project["conditions"])
            print(v)

        break
