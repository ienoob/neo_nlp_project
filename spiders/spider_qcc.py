#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

import json
import random
import numpy as np
from urllib import parse
from spiders.base_spider import get_commom_content
from spiders.selenium_spider import get_tcc_cookie

# url = "https://www.qcc.com/web/search?key=%E6%96%B0%E4%B8%9C%E6%96%B9"
#
# cookie = "zg_did=%7B%22did%22%3A%20%2217d5ad28a09897-06149b8176bee5-2343360-1fa400-17d5ad28a0abf2%22%7D; zg_294c2ba1ecc244809c552f8f6fd2a440=%7B%22sid%22%3A%201642594224670%2C%22updated%22%3A%201642594224675%2C%22info%22%3A%201642594224673%2C%22superProperty%22%3A%20%22%7B%5C%22%E5%BA%94%E7%94%A8%E5%90%8D%E7%A7%B0%5C%22%3A%20%5C%22%E4%BC%81%E6%9F%A5%E6%9F%A5%E7%BD%91%E7%AB%99%5C%22%7D%22%2C%22platform%22%3A%20%22%7B%7D%22%2C%22utm%22%3A%20%22%7B%7D%22%2C%22referrerDomain%22%3A%20%22www.google.com.hk%22%2C%22zs%22%3A%200%2C%22sc%22%3A%200%2C%22firstScreen%22%3A%201642594224670%2C%22cuid%22%3A%20%22undefined%22%7D; qcc_did=cecb6f65-17c3-49b9-a612-4317cbf55017; acw_tc=a3b5239516548537998516876e9060bcad02a3fc05860e0eff9f811995; QCCSESSID=1923da6664c5b8bc0ed913ec86; UM_distinctid=1814cf91d5274a-01405bfde7a2f8-978183a-1fa400-1814cf91d53ed1; CNZZDATA1254842228=1473479896-1637903557-https%253A%252F%252Fwww.google.com%252F%7C1654851405"
# content = get_commom_content(url, cookie=cookie)
#
# print(content)


data_path = "D:\\xxxx\\public_opinion_company_label.json"
# data_path = "D:\\xxxx\\investee_entity.json"
# data_path = "D:\\xxxx\\investor_entity.json"
with open(data_path, "r") as f:
    data_json = f.read()
data_dict = json.loads(data_json)
data_dict_list = [(k, v) for k, v in data_dict.items()]
data_dict_list.sort(key=lambda x: x[1], reverse=True)
np.random.shuffle(data_dict_list)
print(len(data_dict_list))
# i = 0
import os
import time
import pickle
from bs4 import BeautifulSoup
# from spiders.selenium_spider import get_cookie

def spider():
    cookies = "zg_did=%7B%22did%22%3A%20%2217d5ad28a09897-06149b8176bee5-2343360-1fa400-17d5ad28a0abf2%22%7D; zg_294c2ba1ecc244809c552f8f6fd2a440=%7B%22sid%22%3A%201642594224670%2C%22updated%22%3A%201642594224675%2C%22info%22%3A%201642594224673%2C%22superProperty%22%3A%20%22%7B%5C%22%E5%BA%94%E7%94%A8%E5%90%8D%E7%A7%B0%5C%22%3A%20%5C%22%E4%BC%81%E6%9F%A5%E6%9F%A5%E7%BD%91%E7%AB%99%5C%22%7D%22%2C%22platform%22%3A%20%22%7B%7D%22%2C%22utm%22%3A%20%22%7B%7D%22%2C%22referrerDomain%22%3A%20%22www.google.com.hk%22%2C%22zs%22%3A%200%2C%22sc%22%3A%200%2C%22firstScreen%22%3A%201642594224670%2C%22cuid%22%3A%20%22undefined%22%7D; qcc_did=cecb6f65-17c3-49b9-a612-4317cbf55017; acw_tc=a3b5239516548537998516876e9060bcad02a3fc05860e0eff9f811995; QCCSESSID=1923da6664c5b8bc0ed913ec86; UM_distinctid=1814cf91d5274a-01405bfde7a2f8-978183a-1fa400-1814cf91d53ed1; CNZZDATA1254842228=1473479896-1637903557-https%253A%252F%252Fwww.google.com%252F%7C1654851405"
    # print(cookies)
    fail_list = []
    i = 0
    for k, v in data_dict_list:
        # if i < 0 or k[-2:] == "公司" or  "\n" in k or "?" in k or "/" in k or "&" in k or "*" in k or "'" in k or "|" in k:
        #     i += 1
        #     continue
        print("{} start search".format(k))
        path = "F:\download2\\qcc\\{}.html".format(k)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()
            # if item_key == "包大师":
            # print(html)
            soup = BeautifulSoup(html, "html.parser")
            item_list_v1 = soup.find_all("div", {"class": "company"})

            item_list_v2 = soup.find_all("div", {"class": "search-item"})
            if len(item_list_v1)>2 or len(item_list_v2):
                i += 1
                continue

        print("index {}".format(i))
        print(k, v)

        urlcode = parse.urlencode({"key": k})
        url = "https://www.qcc.com/web/search?{}".format(urlcode)

        # r = requests.get(url, headers=HEADER)

        content = get_commom_content(url, cookie=cookies)
        if content is None:
            break
        if k not in content:
            print("get_data: {}".format(len(os.listdir("F:\download2\\qcc\\"))))
            break
            # print("refresh cookie")
            # cookies = get_tcc_cookie()
            #
            # continue
            # break
        # soup = BeautifulSoup(content, "html.parser")
        # item_list_v1 = soup.find_all("div", {"class": "company"})
        #
        # item_list_v2 = soup.find_all("div", {"class": "search-item"})
        # print(len(item_list_v1), len(item_list_v2))
        # if len(item_list_v1) <= 2 and len(item_list_v2)==0:
        #     fail_list.append(k)
        #     print("{} fail search".format(k))
        #     fail_dump = pickle.dumps(fail_list)
        #     with open("D:\\xxxx\\fail_list.txt", "wb") as f:
        #         f.write(fail_dump)
        #     time.sleep(random.randint(12, 24))
        #     continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("{} complete search".format(k))
        i += 1
        time.sleep(random.randint(12, 24))


if __name__ == "__main__":
    spider()
