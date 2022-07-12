#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import time
import hashlib
import numpy as np
from bs4 import BeautifulSoup
from spiders.base_spider import get_commom_content

def spider():
    count = 24345
    page = count//24
    if count%24==0:
        page += 1
    for i in range(1015, page+2):
        url = "https://zdb.pedaily.cn/inv/p{}/".format(i)

        # path = "F:\download2\\zdb_inv\\inv_p_{}.html".format(i)

        content = get_commom_content(url)

        file_md5_indx = hashlib.md5(content.encode()).hexdigest()

        path = "F:\download2\\zdb_inv\\{}.html".format(file_md5_indx)
        if os.path.exists(path):
            time.sleep(20)
            continue
        soup = BeautifulSoup(content, "html.parser")
        item_list = soup.find("ul", {"id": "inv-list"})
        if item_list is None:
            time.sleep(10)
            continue
        item_list = item_list.find_all("li")
        if len(item_list) > 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            sleep_time = np.random.randint(10, 50)
            print(i, sleep_time, url, len(item_list))
            time.sleep(sleep_time)



if __name__ == "__main__":
    spider()
