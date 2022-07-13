#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/4/29 12:11
    @Author  : jack.li
    @Site    : 
    @File    : search_bing.py

"""
import json
import random
import requests

user_agent = [
	"Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)",
	"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
	"Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
	"Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
	"Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
	"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
	"Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
	"Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
	"Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
	"Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
	"Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
	"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
	"Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52"
]

HEADER = {
    'User-Agent': random.choice(user_agent),  # 浏览器头部
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', # 客户端能够接收的内容类型
    'Accept-Language': 'en-US,en;q=0.5', # 浏览器可接受的语言
    'Connection': 'keep-alive', # 表示是否需要持久连接

    "cookie": "MUID=28906E881E4C6B5F270E60151F626A35; ANON=A=99C3BB5718EFF4C5808204A6FFFFFFFF&E=17e1&W=1; MUIDB=28906E881E4C6B5F270E60151F626A35; _tarLang=default=en; _TTSS_IN=hist=WyJlbiIsImphIiwiemgtSGFucyIsImF1dG8tZGV0ZWN0Il0=; _TTSS_OUT=hist=WyJqYSIsInpoLUhhbnMiLCJlbiJd; SRCHD=AF=ANAB01; SRCHUID=V=2&GUID=03925FFEE47C4A5C9D53F7598A94B540&dmnchg=1; imgv=flts=20220419; USRLOC=HS=1; _UR=QS=0&TQS=0; _HPVN=CS=eyJQbiI6eyJDbiI6NSwiU3QiOjAsIlFzIjowLCJQcm9kIjoiUCJ9LCJTYyI6eyJDbiI6NSwiU3QiOjAsIlFzIjowLCJQcm9kIjoiSCJ9LCJReiI6eyJDbiI6NSwiU3QiOjAsIlFzIjowLCJQcm9kIjoiVCJ9LCJBcCI6dHJ1ZSwiTXV0ZSI6dHJ1ZSwiTGFkIjoiMjAyMi0wNC0yMlQwMDowMDowMFoiLCJJb3RkIjowLCJHd2IiOjAsIkRmdCI6bnVsbCwiTXZzIjowLCJGbHQiOjAsIkltcCI6Mn0=; SUID=A; ZHCHATWEAKATTRACT=TRUE; ABDEF=V=13&ABDV=11&MRNB=1651641336986&MRB=0; _EDGE_S=SID=349BDD8BA4936C48116DCC11A5F56D18; _SS=PC=HCTS&SID=349BDD8BA4936C48116DCC11A5F56D18; SRCHS=PC=HCTS; ZHCHATSTRONGATTRACT=TRUE; SRCHUSR=DOB=20200330&T=1651672304000; ipv6=hit=1651675907578&t=4; SRCHHPGUSR=SRCHLANG=zh-Hans&BZA=0&BRW=NOTP&BRH=M&CW=662&CH=754&SW=1536&SH=864&DPR=1.25&UTC=480&DM=0&EXLTT=31&HV=1651673147&WTS=63787269105&PV=10.0.0; SNRHOP=TS=637872699624292327&I=1; _dd_s=logs=1&id=841f53cf-41eb-43dc-a9fa-388a22befce4&created=1651671827950&expire=1651674226070",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="100", "Microsoft Edge";v="100"',
    "sec-ch-ua-arch": "x86",
    "sec-ch-ua-bitness": "64",
    "sec-ch-ua-full-version": "100.0.1185.50",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": "",
    "sec-ch-ua-platform": "Windows",
    "sec-ch-ua-platform-version": "10.0.0",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": None,
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
#     "x-client-data": ‘eyIxIjoiMiIsIjIiOiIxIiwiMyI6IjAiLCI0IjoiNTU3MDc3MDU1MDc3NTc2NjkwNCIsIjUiOiJcIjBCQ2lRYjRObk5VeGJyYkFIanVSYUc1cmgrTVRMZmZLMHdGODhsQ2pNR1E9XCIiLCI2Ijoic3RhYmxlIiwiNyI6IjExMTY2OTE0OTY5NjQiLCI5IjoiZGVza3RvcCJ9
# 已解码:
# message ClientVariations {
#   {"1":"2","2":"1","3":"0","4":"5570770550775766904","5":"\"0BCiQb4NnNUxbrbAHjuRaG5rh+MTLffK0wF88lCjMGQ=\"","6":"stable","7":"1116691496964","9":"desktop"}
# }’

}
import time
from urllib import parse
from bs4 import BeautifulSoup
data_path = "D:\\xxxx\\investee.json"
data_path = "D:\\xxxx\\po_entity.json"
data_path = "D:\\xxxx\\investee_entity.json"
data_path = "D:\\xxxx\\investor_entity.json"
with open(data_path, "r") as f:
    data_json = f.read()

# with open(data_path, "r") as f:
#     data_json = f.read()
data_dict = json.loads(data_json)
print(len(data_dict))
data_dict_list = [(k, v) for k, v in data_dict.items()]
data_dict_list.sort(key=lambda x: x[1], reverse=True)

def spider_one(input_str):
    urlcode = parse.urlencode({"q": input_str})
    url = "https://cn.bing.com/search?{}".format(urlcode)
    r = requests.get(url, headers=HEADER)
    content = r.content.decode()

    print(content)

def spider():
    i = 0
    import os
    for k, v in data_dict_list:
        if k[-2:] == "公司" or  "\n" in k or "?" in k or "/" in k or "&" in k or "*" in k or "'" in k or "|" in k or k[-1] == "-":
            i += 1
            continue
        print("{} start".format(k))
        path = "D:\download\\bing\\{}.html".format(k)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            item_list = soup.find_all("li", {"class": "b_algo"})
            if len(item_list):
                continue
        # print("index {}".format(i))
        # print(k, v)
        # k = "吉利汽车"
        urlcode = parse.urlencode({"q": k})
        url = "https://cn.bing.com/search?{}".format(urlcode)
        r = requests.get(url, headers=HEADER)
        content = r.content.decode("utf-8")
        # print(content)
        # break
        # print(k in content)
        if k not in content:
            break
        soup = BeautifulSoup(content, "html.parser")
        item_list = soup.find_all("li", {"class": "b_algo"})
        if len(item_list)==0:
            print("{} fail".format(k))
            time.sleep(random.randint(12, 24))
            continue

        with open("D:\download\\bing\{}.html".format(k), "w", encoding="utf-8") as f:
            f.write(content)
        print("{} complete".format(k))
        i += 1
        time.sleep(random.randint(12, 24))

def spider_v2():
    from selenium import webdriver

    from selenium.webdriver.chrome.options import Options
    # 声明调用哪个浏览器，本文使用的是Chrome，其他浏览器同理。有如下两种方法及适用情况
    # driver = webdriver.Chrome()#把Chromedriver放到python安装路径里
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(
        executable_path="D:\soft_package\chromedriver_win32\chromedriver.exe", chrome_options=chrome_options)  # 没有把Chromedriver放到python安装路径

    for k, v in data_dict_list:
        print("{} start".format(k))
        if k[-2:] == "公司" or  "\n" in k or "?" in k or "/" in k or "&" in k or "*" in k or "'" in k or "|" in k or k[-1] == "-":
            continue
        path = "D:\download\\bing\\{}.html".format(k)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            item_list = soup.find_all("li", {"class": "b_algo"})
            if len(item_list):
                continue

        url = "https://cn.bing.com/search?q={}".format(k)

        driver.get(url)
        time.sleep(1)

        items = driver.find_elements_by_xpath('//*[@id="b_results"]/li/div/h2/a')
        if len(items):
            print("{} complete".format(k))
            with open("D:\download\\bing\{}.html".format(k), "w", encoding="utf-8") as f:
                f.write(driver.page_source)

        time.sleep(2)
        driver.close()  # 浏览器可以同时打开多个界面，close只关闭当前界面，不退出浏览器
    driver.quit()
import os
def parse_v2():
    from bs4 import BeautifulSoup

    path = "D:\download\\bing\\"

    baike_alias = dict()
    is_alias = 0
    for file in os.listdir(path):
        # print(file)
        # file = "100课堂.html"
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        print(file)
        # if item_key == "包大师":
        print(html)
        soup = BeautifulSoup(html, "html.parser")
        item_list = soup.find_all("li", {"class": "b_algo"})
        for item in item_list:
            print(item.find("h2").text)
        break
        # for item in item_list:
        #     print(item)

if __name__ == "__main__":
    # spider_one("100课堂")
    # parse_v2()
    spider()