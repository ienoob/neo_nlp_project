#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/21 13:55
    @Author  : jack.li
    @Site    : 
    @File    : jin10_spider.py

"""
import os
import time
from spiders.base_spider import get_commom_content
from bs4 import BeautifulSoup


for i in range(1, 958):
    print("page {} start".format(i))
    url = "https://xnews.jin10.com/page/{}".format(i)

    content = get_commom_content(url)

    # print(content)

    soup = BeautifulSoup(content, "html.parser")
    item_list = soup.find("div", {"class": "jin10-news-list"}).find_all("div", {"class": "jin10-news-list-item news"})

    for item in item_list:
        item_url = item.find("a")["href"]
        print(item_url)
        if "https://www.chinanews.com.cn" in item_url:
            continue
        # item_url = item_url.replace("=", "_")
        item_id = item_url.split("/")[-1]
        item_id = item_id.replace("=", "_").replace("?", "_")
        file = "G:\\download2\\jin10\\{}.html".format(item_id)
        if os.path.exists(file):
            continue
        content = get_commom_content(item_url)
        time.sleep(20)
        if len(content) is None:
            continue

        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
    time.sleep(20)

# for idx in range(10405, )