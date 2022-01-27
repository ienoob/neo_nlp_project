#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import requests
import hashlib
import numpy as np

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}

for i in range(10000, 100000):
    url = "https://www.trjcn.com/org/detail_{}.html".format(i)

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        # proxies = {
        #     "http": "1.1.1.1:9090",
        #     "https": "1.1.1.1:9091"
        # }
        data = r.content.decode("utf-8")
        file_md5_indx = hashlib.md5(data.encode()).hexdigest()
        with open("invest_data/{}.html".format(file_md5_indx), "w", encoding="utf-8") as f:
            f.write(data)
    elif r.status_code == 403:
        break

    sleep_time = np.random.randint(10, 100)
    print(sleep_time, url)
    time.sleep(sleep_time)

