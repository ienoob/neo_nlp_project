#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import time
from bs4 import BeautifulSoup
from spiders.base_spider import get_commom_content

# url =
cookie = "UM_distinctid=17e70fe8e33799-066128b179d5c6-978183a-1fa400-17e70fe8e34cd3; CNZZDATA1276128893=40874618-1642570191-https%253A%252F%252Fwww.google.com.hk%252F%7C1642570191; Hm_lvt_af7e7be25931d98caaeb27504df97116=1656059609; PHPSESSID=c90406ad83e3ebee417c472bce8b3f28; app_user=JD1zGwEzX5OJYpHp2do75qK%2BcZTk1S3fOqouqtdh4UgWfG%2F0u6ZPceoccM41nMWHwTdyGSihJwjNBudkGn4Yumt6ym%2B8qng%2BSLkUnwWAQDkU%2Fbrv%2FqkZtcc%2BikVyFBG0; Hm_lpvt_af7e7be25931d98caaeb27504df97116=1656292657"

#
# print(content)

for i in range(1, 7):
    path = "F:\\download2\\vcxpe_inv\\page_nw_{}.html".format(i)
    if os.path.exists(path):
        continue

    url = "https://vcxpe.com/bigdata/finan-event/p{}.html".format(i)


    content = get_commom_content(url, cookie=cookie)

    soup = BeautifulSoup(content, "html.parser")

    items_list = soup.find("tbody")
    if items_list is None:
        print("refresh cookie")
        break
    items_list = items_list.find_all("tr")
    if len(items_list) > 0:
        with open("F:\\download2\\vcxpe_inv\\page_{}.html".format(i), "w", encoding="utf-8") as f:
            f.write(content)
        print("page {} complete!".format(i))
    else:
        print("refresh cookie")
        break

    time.sleep(20)

# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains

# driver = webdriver.Chrome(
# executable_path="D:\\xxxx\\chromedriver_win32_96\\chromedriver.exe")  # 没有把Chromedriver放到python安装路径
#
# url = "https://vcxpe.com/bigdata/finan-event/p100.html"
#
# driver.get(url)
