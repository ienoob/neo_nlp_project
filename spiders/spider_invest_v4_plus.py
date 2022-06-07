#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import time
import requests
import hashlib
import numpy as np
from bs4 import BeautifulSoup

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}

path = "invest_data4"
for i, file in enumerate(os.listdir(path)):
    if i < 764:
        continue
    print("data idx {}".format(i))
    file_path = path + "//" + file

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    soup = BeautifulSoup(data, "lxml")
    subs = soup.find_all(name="tr", attrs={"class": "table_bg1"})
    for item in subs:
        url = item.find(name="td").find("a")["href"]
        r = requests.get(url, headers=headers)

        if r.status_code == 200:
            data = r.content.decode("utf-8")
            file_md5_indx = hashlib.md5(data.encode()).hexdigest()
            with open("invest_data4_plus/{}.html".format(file_md5_indx), "w", encoding="utf-8") as f:
                f.write(data)
        elif r.status_code == 403:
            break

        sleep_time = np.random.randint(10, 100)
        print(sleep_time, url)
        time.sleep(sleep_time)

    subs = soup.find_all(name="tr", attrs={"class": "table_bg2"})
    for item in subs:
        if item.find(name="td") is None:
            continue
        if item.find(name="td").find("a") is None:
            continue
        url = item.find(name="td").find("a")["href"]
        r = requests.get(url, headers=headers)

        if r.status_code == 200:
            data = r.content.decode("utf-8")
            file_md5_indx = hashlib.md5(data.encode()).hexdigest()
            with open("invest_data4_plus/{}.html".format(file_md5_indx), "w", encoding="utf-8") as f:
                f.write(data)
        elif r.status_code == 403:
            break

        sleep_time = np.random.randint(10, 100)
        print(sleep_time, url)
        time.sleep(sleep_time)


