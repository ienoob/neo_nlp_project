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

def get_cookie():
    # 声明调用哪个浏览器，本文使用的是Chrome，其他浏览器同理。有如下两种方法及适用情况
    # driver = webdriver.Chrome()#把Chromedriver放到python安装路径里
    driver = webdriver.Chrome( executable_path="D:\\xxxx\\chromedriver_win32\\chromedriver.exe")#没有把Chromedriver放到python安装路径

    url = "https://www.tianyancha.com/search?key=%E5%8C%97%E4%BA%AC%E6%9C%9D%E5%AE%89%E5%8C%BB%E5%AD%A6"

    driver.get(url)
    cookies = {}
    for cookie in driver.get_cookies():
        cookies[cookie["name"]] = cookie["value"]

    time.sleep(10)
    driver.close()#浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
    driver.quit()#退出整个浏览器

    return cookies



def get_wdj_cookie():
    # 声明调用哪个浏览器，本文使用的是Chrome，其他浏览器同理。有如下两种方法及适用情况
    # driver = webdriver.Chrome()#把Chromedriver放到python安装路径里
    driver = webdriver.Chrome( executable_path="D:\\xxxx\\chromedriver_win32_96\\chromedriver.exe")#没有把Chromedriver放到python安装路径

    url = "https://www.wandoujia.com/"

    driver.get(url)
    cookies = {}
    for cookie in driver.get_cookies():
        cookies[cookie["name"]] = cookie["value"]

    time.sleep(10)
    driver.close()#浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
    driver.quit()#退出整个浏览器

    return cookies

def get_crunchbase_cookie():
    # 声明调用哪个浏览器，本文使用的是Chrome，其他浏览器同理。有如下两种方法及适用情况
    # driver = webdriver.Chrome()#把Chromedriver放到python安装路径里
    driver = webdriver.Chrome( executable_path="D:\\xxxx\\chromedriver_win32_96\\chromedriver.exe")#没有把Chromedriver放到python安装路径

    url = "https://www.crunchbase.com/organization/kkday"

    driver.get(url)
    cookies = {}
    for cookie in driver.get_cookies():
        cookies[cookie["name"]] = cookie["value"]

    time.sleep(20)
    driver.close()#浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
    driver.quit()#退出整个浏览器

    return cookies

def get_tcc_cookie():
    # 声明调用哪个浏览器，本文使用的是Chrome，其他浏览器同理。有如下两种方法及适用情况
    # driver = webdriver.Chrome()#把Chromedriver放到python安装路径里
    driver = webdriver.Chrome( executable_path="D:\\xxxx\\chromedriver_win32_96\\chromedriver.exe")#没有把Chromedriver放到python安装路径

    url = "https://www.qcc.com"

    driver.get(url)
    cookies = {}
    for cookie in driver.get_cookies():
        cookies[cookie["name"]] = cookie["value"]

    time.sleep(10)
    driver.close()#浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
    driver.quit()#退出整个浏览器

    return cookies


if __name__ == "__main__":
    cooki = get_tcc_cookie()
    print(cooki)
