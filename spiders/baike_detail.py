#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/16 14:17
    @Author  : jack.li
    @Site    : 
    @File    : baike_detail.py

"""
import os
import time
import random
import hashlib
from bs4 import BeautifulSoup
from spiders.base_spider import get_commom_content

def spider():
    path = "F:\download2\\baike\\"
    for file in os.listdir(path):
        # file = "薇雅.html"
        item_key = file[:-5]
        # item_key = "李家琦"
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        # print(html)
        # if item_key == "包大师":
        # print(html)
        soup = BeautifulSoup(html, "html.parser")
        item_list = soup.find("dl", {"class": "search-list"})
        if item_list is None:
            continue
        item_list = item_list.find_all("dd")
        for item in item_list:

            # if item.a["class"]
            if "result-title" not in  item.a["class"]:
                continue
            url = item.a["href"]
            if url[:5] == "/item":
                url = "https://baike.baidu.com" + url
            url_id = hashlib.md5(url.encode("utf-8")).hexdigest()
            save_path = "F:\download2\\baidubaike\\{}.html".format(url_id)

            if os.path.exists(save_path):
                continue
            print(item.a.text)
            print(item.p.text)

            content = get_commom_content(url)
            if content is None:
                continue

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)

            time.sleep(random.randint(5, 15))
            # if "公司" in item.text:
            #     print(item.text.split(" ")[0])
            #     baike_alias[item_key].append(item.text.split(" ")[0])


if __name__ == "__main__":
    spider()
