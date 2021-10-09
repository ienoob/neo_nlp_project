#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import os
import json
from pytorch.other_info_extraction.policy_information_extractor_v2 import Document

if __name__ == "__main__":
    path = "D:\data\政策信息抽取\\text"
    file_list = os.listdir(path)
    print(len(file_list))

    doc = Document()
    for file in file_list[64:]:
        print(file)
        file_path = path + "\\" + file
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        print([data])
        doc.parse_content(data)
        doc.display_document()
        res = doc.parse_half_struction()
        print(json.dumps(res, ensure_ascii=False, indent=4))

        doc.parse_precision(res["conditions"])
        doc.parse_contact_info()

        for k_project in res["project_infos"]:
            print("项目名称", k_project["project_name"])
            doc.parse_precision(k_project["conditions"])


        break
