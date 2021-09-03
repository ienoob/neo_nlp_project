#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
from bs4 import BeautifulSoup
from pytorch.other_info_extraction.policy_data import item_list
from pytorch.other_info_extraction.policy_information_extractor import Document


if __name__ == "__main__":
    doc = Document()
    # doc.parse_content(content2)
    # doc.extract_zb()
    # res = mrc("服务收入占比", "1、申报主体需获评省级服务型制造示范企业，服务收入占企业营业收入比重达30%以上；")
    # print(res)

    for item in item_list:
        file_name = "D:\data\\政策信息抽取\\{}.txt".format(item[1])

        with open(file_name, "rb") as f:
            data = f.read()

        soup = BeautifulSoup(data, 'html.parser')
        doc.parse_content(soup.text)
        res = doc.extract_zb()
        for target_info in res:
            print(json.dumps(target_info, indent=4, ensure_ascii=False))

    # print(json.dumps({"name": "分类与排序 "}, indent=4, ensure_ascii=False))
