#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/4/29 0:09
    @Author  : jack.li
    @Site    : 
    @File    : tyc_search.py

"""
import numpy as np
import json
import random
import requests
from spiders.base_spider import get_content
import pandas as pd



from urllib import parse
data_path = "D:\\xxxx\\investee.json"
data_path = "D:\\xxxx\\po_entity.json"
data_path = "D:\\xxxx\\invest_new.json"
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
from spiders.selenium_spider import get_cookie

def spider():
    cookies = get_cookie()
    fail_list = []
    i = 0
    for k, v in data_dict_list:
        if i < 0 or k[-2:] == "公司" or  "\n" in k or "?" in k or "/" in k or "&" in k or "*" in k or "'" in k or "|" in k:
            i += 1
            continue
        print("{} start search".format(k))
        path = "F:\download2\\tyc\\{}.html".format(k)
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
        url = "https://www.tianyancha.com/search?{}".format(urlcode)

        # r = requests.get(url, headers=HEADER)

        content = get_content(url, cookies=cookies)
        if content is None:
            continue
        if k not in content:
            print("refresh cookie")
            cookies = get_cookie()
            print("get_data: {}".format(len(os.listdir("F:\download2\\tyc\\"))))
            continue
            # break
        soup = BeautifulSoup(content, "html.parser")
        item_list_v1 = soup.find_all("div", {"class": "company"})

        item_list_v2 = soup.find_all("div", {"class": "search-item"})
        print(len(item_list_v1), len(item_list_v2))
        if len(item_list_v1) <= 2 and len(item_list_v2)==0:
            fail_list.append(k)
            print("{} fail search".format(k))
            fail_dump = pickle.dumps(fail_list)
            with open("D:\\xxxx\\fail_list.txt", "wb") as f:
                f.write(fail_dump)
            time.sleep(random.randint(12, 24))
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("{} complete search".format(k))
        i += 1
        time.sleep(random.randint(12, 24))



def spider_one(name):
    urlcode = parse.urlencode({"key": name})
    url = "https://www.tianyancha.com/search?{}".format(urlcode)

    # r = requests.get(url, headers=HEADER)

    content = get_content(url)

    print(content)

    soup = BeautifulSoup(content, "html.parser")
    item_list_v1 = soup.find_all("div", {"class": "company"})

    item_list_v2 = soup.find_all("div", {"class": "search-item"})

    print(len(item_list_v1), len(item_list_v2))

def parse_v2():
    from bs4 import BeautifulSoup

    path = "G:\download2\\tyc\\"

    tyc_alias = dict()
    is_alias = 0
    i = 0
    for file in os.listdir(path):
        item_key = file[:-5]
        # print(file)
        # "search-item sv-search-company  "
        # file = "Manner.html"
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        print(file)
        # if item_key == "包大师":
        # print(html)
        soup = BeautifulSoup(html, "html.parser")


        item_list_v2 = soup.find_all("div", {"class": "search-item"})
        for item in item_list_v2:
            company_name = item.find("div", {"class": "info"}).text.strip()
            print(company_name, "search item")
            tyc_alias.setdefault(company_name, [])
            tyc_alias[company_name].append(item_key)

            break


        # for item in item_list:
        #     print(item.find("div", {"class": "info"}).text)
        item_list_v1 = soup.find_all("div", {"class": "company"})
        # item_list = soup.find_all("div", {"class": "company"})
        # print(len(item_list))
        for item in item_list_v1:
            item = item.find("span", {"class": "left"})
            if not item:
                continue
            i += 1
            company_name = item.text.strip()
            print(company_name, "company")
            tyc_alias.setdefault(company_name, [])
            tyc_alias[company_name].append(item_key)
            break
        item_list = soup.find_all("div", {"class": "middle"})
        for item in item_list:
            if item.text[:5] == "所属公司：":
                # print()
                company_name = item.text[5:]

                tyc_alias.setdefault(company_name, [])
                tyc_alias[company_name].append(item_key)
                i += 1
                break
    #     print(i)
    #     # print(tyc_alias)
        print(len(tyc_alias))

    with open("D:\\xxxx\\tyc_search_alias.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(tyc_alias))

def single_one():
    urlcode = parse.urlencode({"key": "包小盒"})
    url = "https://www.tianyancha.com/search?{}".format(urlcode)
    content = get_content(url)
    # print(content)

    soup = BeautifulSoup(content, "html.parser")
    item_list_v1 = soup.find_all("div", {"class": "company"})

    item_list_v2 = soup.find_all("div", {"class": "search-item"})

    print(len(item_list_v1), len(item_list_v2))


def fail_spider():
    with open("D:\\xxxx\\fail_list.txt", "rb") as f:
        data = f.read()
    items = pickle.loads(data)
    np.random.shuffle(items)
    print("items {}".format(len(items)))
    for item in items:
        print("search {}".format(item))
        urlcode = parse.urlencode({"key": item})
        url = "https://www.tianyancha.com/search?{}".format(urlcode)
        content = get_content(url)

        if content is None:
            continue

        soup = BeautifulSoup(content, "html.parser")
        item_list_v1 = soup.find_all("div", {"class": "company"})

        item_list_v2 = soup.find_all("div", {"class": "search-item"})
        print(len(item_list_v1), len(item_list_v2))

        if len(item_list_v1) <= 2 and len(item_list_v2) == 0:
            print("{} fail".format(item))
            time.sleep(random.randint(12, 24))
            continue
        with open("G:\download2\\tyc\\{}.html".format(item), "w", encoding="utf-8") as f:
            f.write(content)
        print("{} complete".format(item))
        time.sleep(random.randint(12, 24))

if __name__ == "__main__":
    # spider_one("Outer")
    # parse_v2()
    # fail_spider()
    # parse_v2()
    spider()
