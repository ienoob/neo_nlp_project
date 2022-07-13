#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/26 10:05
    @Author  : jack.li
    @Site    : 
    @File    : sohu_spider.py

"""
import pandas as pd
from bs4 import BeautifulSoup
from base_spider import get_commom_content

url = "https://www.sohu.com/na/466484009_120056153"
content = get_commom_content(url)
# print(content)

soup = BeautifulSoup(content, "html.parser")
world2000 = []
items = soup.find("table").find_all("tr")
for item in items:
    td_list = item.find_all("td")
    if len(td_list) == 3:
        print(td_list[1].text)
        world2000.append((td_list[0].text, td_list[1].text, td_list[2].text))
world2000_df = pd.DataFrame(world2000, columns=["rank", "name", "country"])
world2000_df.to_csv("D:\\xxxx\\world2000.csv", index=False)
# print(len(world1000))
