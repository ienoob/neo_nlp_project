#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import hashlib
import numpy as np
from spiders.base_spider import get_commom_content

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}

for i in range(920, 1501):
    # 排序方式 0， 1， 2， 3
    url = "https://org.pedata.cn/list_{}_0_0_0_0_3.html".format(i)
    print(url)
    content = get_commom_content(url)
    if content is None:
        continue
    if content == 403:
        break
    file_md5_indx = hashlib.md5(content.encode()).hexdigest()
    with open("F:\download2\invest_data4\{}.html".format(file_md5_indx), "w", encoding="utf-8") as f:
        f.write(content)

    sleep_time = np.random.randint(10, 100)
    print(sleep_time, url)
    time.sleep(sleep_time)
