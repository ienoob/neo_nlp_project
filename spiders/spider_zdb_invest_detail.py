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
    path = "F:\\download2\\zdb_inv\\"

    for file in os.listdir(path):
        file_path = path + file

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")
        item_list = soup.find("ul", {"id": "inv-list"})
        if item_list is None:
            continue
        item_list = item_list.find_all("li")
        for item in item_list:

            inv_detail_url = item.find("div", {"class": "view"})
            if inv_detail_url is None:
                continue
            inv_detail_url = inv_detail_url.find("a")
            inv_detail_url = "https://zdb.pedaily.cn"+inv_detail_url["href"]

            file_md5_indx = hashlib.md5(inv_detail_url.encode()).hexdigest()

            save_path = "F:\download2\\zdb_inv_detail\\{}.html".format(file_md5_indx)
            if os.path.exists(save_path):
                continue

            content = get_commom_content(inv_detail_url)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)

            sleep_time = np.random.randint(10, 50)
            print(sleep_time, inv_detail_url)
            time.sleep(sleep_time)

        # break


if __name__ == "__main__":
    spider()
