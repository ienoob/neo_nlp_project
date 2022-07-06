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

url = "https://gs.amac.org.cn/amac-infodisc/res/pof/manager/index.html"

driver.get(url)

driver.maximize_window()
time.sleep(10)

button = driver.find_element(by=By.XPATH, value='//*[@id="layui-layer1"]/div[3]/a')
button.click()


down_button = driver.find_element(by=By.NAME, value="managerList_length")
ActionChains(driver).move_to_element(down_button).click(down_button).perform()
time.sleep(1)

option_list = down_button.find_elements(by=By.TAG_NAME, value="option")
# print(option_list)
option_list[3].click()

time.sleep(2)

items_row = []
for i in range(245):
    items = driver.find_element(value="managerList").find_elements(by=By.TAG_NAME, value="tr")

    for item in items:
        item_attrs = item.find_elements(by=By.TAG_NAME, value="td")
        row = [ia.text for ia in item_attrs]
        print(row)
        if row:
            item_url = item_attrs[1].find_element(by=By.TAG_NAME, value="a").get_attribute("href")
            row.append(item_url)
            items_row.append(row)
        print(item.text)

    # break
    #
    items_df = pd.DataFrame(items_row, columns=["编号", "私募基金管理人名称", "法定代表人/执行事务合伙人(委派代表)姓名",
                                                "机构类型", "注册地", "登记编号", "成立时间", "登记时间", "url"])
    items_df.to_csv("D:\\xxxx\\amac.csv", index=False)
    time.sleep(5)

    if len(items) == 101:
        next_button = driver.find_element(by=By.XPATH, value='//*[@id="managerList_paginate"]/a[3]')
        ActionChains(driver).move_to_element(next_button).click(next_button).perform()
        # print(next_button.text)
        # next_button.click()
        time.sleep(20)

    print("page {} complete".format(i+1))


#
driver.close()#浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
driver.quit()#退出整个浏览器




