#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/4/10 10:23
    @Author  : jack.li
    @Site    : 
    @File    : spider_convid.py

"""
import re
import requests
from bs4 import BeautifulSoup

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}
for i in range(1, 44):
        url = "http://sh.bendibao.com/news/list_17_623_{}.htm".format(i)
        print("all {}".format(url))
        r = requests.get(url, headers=headers)
        # print(r.content.decode())

        soup = BeautifulSoup(r.content.decode(),  'html.parser')

        items = soup.find_all("a", {"class": "J-share-a"})
        for item in items:

                if re.search("\d月\d{1,2}日上海新增\d+例本土确诊 \d+例无症状", item.text):
                        print(item["href"], item.text)
                elif re.search("\d月\d{1,2}日上海新增本土确诊\d+例 无症状\d+例", item.text):
                        print(item["href"], item.text)
                elif re.search("\d月\d{1,2}日上海新增\d+例本土确诊\d+例无症状", item.text):
                        print(item["href"], item.text)
                elif re.search("\d月\d{1,2}日上海无新增本土确诊新增\d+例本土无症状", item.text):
                        print(item["href"], item.text)
                elif re.search("\d月\d{1,2}日上海新增\d+例本土确诊\+\d+例本土无症状", item.text):
                        print(item["href"], item.text)
                elif re.search("\d月\d{1,2}日上海新增本土\d+\+\d+", item.text):
                        print(item["href"], item.text)