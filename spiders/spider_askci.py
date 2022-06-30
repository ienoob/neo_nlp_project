#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/6 16:41
    @Author  : jack.li
    @Site    : 
    @File    : spider_askci.py

"""
import time
import random
import requests
from spiders.get_proxy import get_proxy
from bs4 import BeautifulSoup
user_agent = [
	"Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)",
	"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
	"Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
	"Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
	"Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
	"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
	"Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
	"Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
	"Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
	"Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
	"Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
	"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
	"Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52"
]
# url = "https://s.askci.com/stock/h/0-0?reportTime=2021-12-31&pageNum=3#QueryCondition"
max_time = 30
proxy_list = get_proxy()
def get_data(url, retry=0):
    if retry>=max_time:
        print("retry out of times")
        return None
    HEADER = {
        'User-Agent': random.choice(user_agent),  # 浏览器头部
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
        'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
        'Connection': 'keep-alive',  # 表示是否需要持久连接
    }
    if len(proxy_list) == 0:
        time.sleep(3)
        p_list = get_proxy()
        for p in p_list:
            proxy_list.append(p)
        if len(proxy_list) == 0:
            print("proxy num empty")
            return None
    ip_proxy = proxy_list.pop(0)
    proxies = {
        'http': ip_proxy,
        'https': ip_proxy,
    }
    try:
        r = requests.get(url, headers=HEADER, proxies=proxies)
    except Exception as e:
        print("retry {}".format(retry))
        return get_data(url, retry+1)
    proxy_list.append(ip_proxy)

    return r.content.decode("utf-8")

import os
def spider():
    for i in range(1, 420):
        print("page {} start".format(i))
        path = "D:\download\\askci\hk_{}.html".format(i)
        path = "D:\download\\askci\\a_{}.html".format(i)
        path = "D:\download\\askci\\xsb_{}.html".format(i)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            soup = BeautifulSoup(content, "html.parser")
            spans = soup.find("div", {"class": "container_12 mg_ttwo"})
            if spans:
                continue

        url = "https://s.askci.com/stock/h/0-0?reportTime=2021-12-31&pageNum={}#QueryCondition".format(i)  # 238
        url = "https://s.askci.com/stock/a/0-0?reportTime=2021-12-31&pageNum={}#QueryCondition".format(i)  # 205
        url = "https://s.askci.com/stock/xsb/0-0?reportTime=2021-12-31&pageNum={}#QueryCondition".format(i) # 419


        res = get_data(url)
        if res is None:
            break

        soup = BeautifulSoup(res, "html.parser")
        spans = soup.find("div", {"class": "container_12 mg_ttwo"})
        if spans is None:
            print(res)
            print("page {} fail".format(i))
            break


        # for span in spans:
        #     print(span.find_all("td")[2].text)
        #     print(span.find_all("td")[3].text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(res)
        print("page {} complete".format(i))
        # time.sleep(random.randint(1, 10))
        #
        # break


        # with open("D:\download\\askci\")


def parse():
    path = "D:\download\\askci\\"

    ss_company_dict = dict()
    for file in os.listdir(path):
        print(file)
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # print(content)
        soup = BeautifulSoup(content, "html.parser")
        spans = soup.find("div", {"class": "container_12 mg_ttwo"}).find("tbody").find_all("tr")
        #
        for span in spans:
            short_name = span.find_all("td")[2].text
            company_name = span.find_all("td")[3].text
            print(company_name, short_name)
            if len(short_name) < 2:
                continue

            if company_name[-2:] != "公司":
                continue

            ss_company_dict.setdefault(company_name, [])
            ss_company_dict[company_name].append(short_name)
            if short_name[-1] == "A":
                ss_company_dict[company_name].append(short_name[:-1])
            if short_name[-2:] == "-B":
                ss_company_dict[company_name].append(short_name[:-2])
            if short_name[-2:] == "-W":
                ss_company_dict[company_name].append(short_name[:-2])

        # break
    import json
    print(ss_company_dict.get("龙版传媒", "&&&"))

    with open("D:\\xxxx\\ss_company_dict.json", "w") as f:
        f.write(json.dumps(ss_company_dict))


if __name__ == "__main__":
    parse()
    # spider()
    # fail

