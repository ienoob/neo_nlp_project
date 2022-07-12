#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome(
executable_path="D:\\xxxx\\chromedriver_win32_96\\chromedriver.exe")  # 没有把Chromedriver放到python安装路径

url = "https://36kr.com/projects?pageNo=1"
driver.get(url)

driver.maximize_window()
time.sleep(10)
items_row = []
for i in range(1500):
    print("page {} start ".format(i))
    items = driver.find_element(by=By.CLASS_NAME, value="ant-table-tbody").find_elements(by=By.TAG_NAME, value="tr")

    for item in items:
        item_attrs = item.find_elements(by=By.TAG_NAME, value="td")
        # print(len(item_attrs))
        row = [ia.text for ia in item_attrs]


        name, desc = row[0].split("\n")
        item_url = item_attrs[0].find_element(by=By.TAG_NAME, value="a").get_attribute("href")
        items_row.append({"name": name,
                          "desc": desc,
                          "行业": row[1],
                          "轮次": row[2],
                          "融资金额": row[3],
                          "地区": row[4],
                          "成立时间": row[5],
                          "url": item_url})
    print("page {} complete".format(i))
    print(len(items))
    if len(items) > 0:
        next_button = driver.find_element(by=By.CLASS_NAME, value='ant-pagination-next')
        ActionChains(driver).move_to_element(next_button).click(next_button).perform()
        # print(next_button.text)
        # next_button.click()
        time.sleep(20)
    else:
        print("page {} is the last page".format(i))
        break

    items_df = pd.DataFrame(items_row, columns=["name", "desc", "行业",
                                                "轮次", "融资金额", "地区", "成立时间", "url"])
    items_df.to_csv("D:\\xxxx\\36kr_project.csv", index=False)
    time.sleep(5)
