#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/4/28 22:29
    @Author  : jack.li
    @Site    : 
    @File    : baike_search.py

"""
import json
from spiders.base_spider import get_commom_content
import time
from urllib import parse
data_path = "D:\\xxxx\\investee.json"
data_path = "D:\\xxxx\\po_entity.json"
# data_path = "D:\\xxxx\\investor_entity.json"
# data_path = "D:\\xxxx\\investee_entity.json"

with open(data_path, "r") as f:
    data_json = f.read()
data_dict = json.loads(data_json)
print(len(data_dict))
data_dict_list = [(k, v) for k, v in data_dict.items()]
data_dict_list.sort(key=lambda x: x[1], reverse=True)
import os

def spider():
    i = 0

    for k, v in data_dict_list:
        if i < 0 or k[-2:] == "公司" or  "\n" in k or "?" in k or "/" in k or "&" in k or "*" in k or "'" in k or "|" in k or ">" in k or "." in k or ":" in k or '"' in k or "<" in k:
            i += 1
            continue
        if k in ["AUX"]:
            i += 1
            continue
        path = "G:\download2\\baike\{}.html".format(k)
        if os.path.exists(path):
            i += 1
            continue
        # with open()
        print("index {}".format(i))
        print(k, v)
        urlcode = parse.urlencode({"word": k})
        url = "https://baike.baidu.com/search/none?{}&pn=0&rn=10&enc=utf8".format(urlcode)

        content = get_commom_content(url)
        if k not in content:
            break
        # print(r.content.decode())
        with open("G:\download2\\baike\{}.html".format(k), "w", encoding="utf-8") as f:
            f.write(content)
        i += 1
        time.sleep(10)


def parse_v2():
    from bs4 import BeautifulSoup

    path = "G:\download2\\baike\\"

    baike_alias = dict()
    is_alias = 0
    for file in os.listdir(path):
        item_key = file[:-5]
        baike_alias[item_key] = []
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        print(file)
        # if item_key == "包大师":
            # print(html)
        soup = BeautifulSoup(html, "html.parser")
        item_list = soup.find_all("a", {"class": "result-title"})
        for item in item_list:
            # print(item.text)
            if "公司" in item.text:
                print(item.text.split(" ")[0])
                baike_alias[item_key].append(item.text.split(" ")[0])
            # break
        if len(baike_alias[item_key]):
            is_alias += 1
    # # break
    print(is_alias, len(baike_alias))

    # file_path = path + "包大师.html"
    # with open(file_path, "r", encoding="utf-8") as f:
    #     html = f.read()
    # soup = BeautifulSoup(html, "html.parser")
    # item_list = soup.find_all("a", {"class": "result-title"})
    # for item in item_list:
    #     print(item.text)


    with open("D:\\xxxx\\baike_search_alias.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(baike_alias))

def spider_():
    i = 0
    d = dict()
    for k, v in data_dict_list:
        if i < 0 or k[
                    -2:] == "公司" or "\n" in k or "?" in k or "/" in k or "&" in k or "*" in k or "'" in k or "|" in k or ">" in k or "." in k or ":" in k or '"' in k or "<" in k:
            i += 1
            continue
        if k in ["AUX"]:
            i += 1
            continue
        path = "G:\download2\\baike\{}.html".format(k)
        if os.path.exists(path):
            i += 1
            d[k] = 1

    with open("D:\\xxxx\\{}.json".format("baike_visit"), "w") as f:
        f.write(json.dumps(d))


if __name__ == "__main__":
    spider_()