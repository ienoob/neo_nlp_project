#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/6/9 11:58
    @Author  : jack.li
    @Site    : 
    @File    : selenium_spider.py

"""

import time
from selenium import webdriver
# 声明调用哪个浏览器，本文使用的是Chrome，其他浏览器同理。有如下两种方法及适用情况
# driver = webdriver.Chrome()#把Chromedriver放到python安装路径里
driver = webdriver.Chrome( executable_path="D:\soft_package\chromedriver_win32\chromedriver.exe")#没有把Chromedriver放到python安装路径

url = "https://cn.bing.com/search?q=100课堂"

driver.get(url)

time.sleep(3)
driver.close()#浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
driver.quit()#退出整个浏览器
