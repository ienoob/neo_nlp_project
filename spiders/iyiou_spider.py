#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/20 22:58
    @Author  : jack.li
    @Site    : 
    @File    : iyiou_spider.py

"""
import os
import time
from bs4 import BeautifulSoup
from spiders.base_spider import get_commom_content

def spider():
    url = "https://www.iyiou.com/news"
    content = get_commom_content(url)
    # print(content)
    soup = BeautifulSoup(content, "html.parser")
    item_list = soup.find("div", {"class": "info-wrap"}).find_all("li")

    for item in item_list:
        item_url = item.find("a")["href"]
        type_id = "{}_{}".format(item_url.split("/")[-2],item_url.split("/")[-1])

        file = "G:\\download2\\iyiou_news\\{}.html".format(type_id)
        if os.path.exists(file):
            continue
        content = get_commom_content(item_url)
        time.sleep(20)
        if len(content) == 0:
            continue
        if "很抱歉，由于您访问的URL有可能对网站造成安全威胁，您的访问被阻断。" in content:
            print("403 error")
            continue

        with open(file, "w", encoding="utf-8") as f:
            f.write(content)

    #     time.sleep(20)

    url = "https://www.iyiou.com/briefing"
    content = get_commom_content(url)
    # print(content)
    soup = BeautifulSoup(content, "html.parser")
    item_list = soup.find("div", {"class": "brief-item"})
    if item_list is None:
        item_list = []
    item_list = item_list.find_all("li")
    for item in item_list:
        item_url = item.find("a")["href"]
        print(item_url)
        type_id = "{}_{}".format(item_url.split("/")[-2], item_url.split("/")[-1])
        file = "G:\\download2\\iyiou_news\\{}.html".format(type_id)
        if os.path.exists(file):
            continue

        full_url = "https://www.iyiou.com" + item_url
        content = get_commom_content(full_url)
        time.sleep(20)
        if len(content) == 0:
            continue
        if "很抱歉，由于您访问的URL有可能对网站造成安全威胁，您的访问被阻断。" in content:
            print("403 warning")
            continue

        with open(file, "w", encoding="utf-8") as f:
            f.write(content)