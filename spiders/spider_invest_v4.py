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

for i in range(1401, 1501):
    # 排序方式 0， 1， 2， 3
    url = "https://org.pedata.cn/list_{}_0_0_0_0_1.html".format(i)
    print(url)
    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        data = r.content.decode("utf-8")
        file_md5_indx = hashlib.md5(data.encode()).hexdigest()
        with open("F:\download2\invest_data4\{}.html".format(file_md5_indx), "w", encoding="utf-8") as f:
            f.write(data)
    elif r.status_code == 403:
        break

    sleep_time = np.random.randint(10, 100)
    print(sleep_time, url)
    time.sleep(sleep_time)
