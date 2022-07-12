#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import time
import hashlib
import numpy as np
import pandas as pd
from spiders.base_spider import get_commom_content


def spider():
    path = "F:\download2\\amac.csv"

    df = pd.read_csv(path)

    for idx, row in df.iterrows():
        print(row["url"])
        url = row["url"]

        file_md5_indx = hashlib.md5(url.encode()).hexdigest()
        #
        save_path = "F:\download2\\amac_detail\\{}.html".format(file_md5_indx)
        if os.path.exists(save_path):
            continue

        content = get_commom_content(url)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        sleep_time = np.random.randint(10, 50)
        print(sleep_time, url)
        time.sleep(sleep_time)

        # break


if __name__ == "__main__":
    spider()
