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

# 33723 到底了
for i in range(33723, 100000):
    url = "https://zdb.pedaily.cn/company/show{}/".format(i)
    r = requests.get(url, headers=headers)
    # print(r.content.decode("utf-8"))

    if r.status_code == 200:
        data = r.content.decode("utf-8")
        file_md5_indx = hashlib.md5(data.encode()).hexdigest()
        with open("invest_data2/{}.html".format(file_md5_indx), "w", encoding="utf-8") as f:
            f.write(data)
    elif r.status_code == 403:
        break

    sleep_time = np.random.randint(10, 100)
    print(sleep_time, url)
    time.sleep(sleep_time)
