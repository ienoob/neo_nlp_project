#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/26 9:28
    @Author  : jack.li
    @Site    : 
    @File    : maigoo_spider.py

"""
import pandas as pd
from bs4 import BeautifulSoup
from spiders.base_spider import get_commom_content

url = "https://www.maigoo.com/news/479347.html"

content = get_commom_content(url)

with open("G:\\download2\\maigoo\\500.html", "w", encoding="utf-8") as f:
    f.write(content)

soup = BeautifulSoup(content, "html.parser")
europe500 = []
items = soup.find("table", {"class": "mod_table table1 fcolor30"}).find_all("tr")
for item in items:
    td_list = item.find_all("td")
    if len(td_list) == 5:
        print(td_list[1].text)
        europe500.append((td_list[0].text, td_list[1].text, td_list[2].text, td_list[3].text, td_list[4].text))

print(len(europe500))

# europe500_df = pd.DataFrame(europe500, columns=[""])
# print(content)