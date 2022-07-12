#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from bs4 import  BeautifulSoup
from spiders.base_spider import get_commom_content

# 美股信息


def spider():

    url = "http://vip.stock.finance.sina.com.cn/usstock/ustotal.php"

    content = get_commom_content(url, decoding="gbk")

    with open("F:\download2\\sina_america_stock.html", "w", encoding="utf-8") as f:
        f.write(content)
    # # print(content)
    #
    # soup = BeautifulSoup(content, "html.parser")
    # it_count = 0
    # items = soup.find_all("div", {"class": "col_div"})
    # for item in items:
    #     for it_url in item.find_all("a"):
    #         print(it_url)
    #         it_count += 1
    # print(it_count)

    url = "http://quote.eastmoney.com/usstocklist.html"

    content = get_commom_content(url, decoding="gbk")
    with open("F:\download2\\eastmony_america_stock.html", "w", encoding="utf-8") as f:
        f.write(content)

    soup = BeautifulSoup(content, "html.parser")

    item = soup.find("div", {"id": "quotesearch"})
    iv_count = 0
    item_list = item.find_all("a")
    for item in item_list:
        print(item, iv_count)
        iv_count += 1




if __name__ == "__main__":
    spider()
