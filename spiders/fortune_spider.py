#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/25 23:55
    @Author  : jack.li
    @Site    : 
    @File    : fortune_spider.py

"""
import pandas as pd
from bs4 import BeautifulSoup
from spiders.base_spider import get_commom_content

url = "https://www.fortunechina.com/fortune500/c/2021-06/02/content_390831.htm"
content = get_commom_content(url)
# print(content)
soup = BeautifulSoup(content, "html.parser")

usa_500 = []
items = soup.find("table", {"id": "table1"}).find_all("tr")
print(len(items))
for item in items:
    td_list = item.find_all("td")
    print(len(td_list))
    if len(td_list) == 4:
        usa_500.append((td_list[0].text, td_list[1].text, td_list[2].text, td_list[3].text))
    print("+++++++++++++++++++++")
    # print(item)
    # print(td_list[1].text)

usa_500_df = pd.DataFrame(usa_500, columns=["rank", "name", "income", "profile"])
usa_500_df.to_csv("D:\\xxxx\\usa_500.csv", index=False)


url = "https://www.fortunechina.com/fortune500/c/2021-08/02/content_394571.htm"

content = get_commom_content(url)
# print(content)
soup = BeautifulSoup(content, "html.parser")

world_500 = []
items = soup.find("table", {"id": "table1"}).find_all("tr")
print(len(items))
for item in items:
    td_list = item.find_all("td")
    print(len(td_list))
    if len(td_list) == 6:
        world_500.append((td_list[0].text, td_list[1].text, td_list[2].text, td_list[3].text, td_list[4].text, td_list[5].text))
    print("+++++++++++++++++++++")
    # print(item)
    # print(td_list[1].text)

usa_500_df = pd.DataFrame(world_500, columns=["rank", "name", "income", "profile", "country", "unk"])
usa_500_df.to_csv("D:\\xxxx\\world_500.csv", index=False)