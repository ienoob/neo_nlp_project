#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/23 23:12
    @Author  : jack.li
    @Site    : 
    @File    : pedaily_spider.py

"""
import os
import hashlib
import time
from spiders.base_spider import get_commom_content

def spider_people():
    for i in range(1, 10000):
        print("index {} start".format(i))
        url = "https://zdb.pedaily.cn/people/show{}/".format(i)
        file_md5_indx = hashlib.md5(url.encode()).hexdigest()
        file = "G:\\download2\\pedaily_people\\{}.html".format(file_md5_indx)
        if os.path.exists(file):
            continue
        content = get_commom_content(url)


        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        time.sleep(10)
        print("index {} complete".format(i))

def spider_project():
    for i in range(1, 10000):
        print("index {} start".format(i))
        url = "https://newseed.pedaily.cn/data/project/{}".format(i)
        file_md5_indx = hashlib.md5(url.encode()).hexdigest()
        file = "G:\\download2\\pedaily_project\\{}.html".format(file_md5_indx)
        if os.path.exists(file):
            continue
        content = get_commom_content(url)

        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
        time.sleep(10)
        print("index {} complete".format(i))

if __name__ == "__main__":
    spider_project()