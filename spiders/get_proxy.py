#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/6 20:56
    @Author  : jack.li
    @Site    : 
    @File    : get_proxy.py

"""
import random
import time

import requests
from bs4 import BeautifulSoup


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
import time

def proxy_v1():
	proxy_list = set()
	url_list = ["https://www.kuaidaili.com/free/inha/1/", "https://www.kuaidaili.com/free/intr/1/"]
	for url in url_list:
		HEADER = {
			'User-Agent': random.choice(user_agent),  # 浏览器头部
			'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
			'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
			'Connection': 'keep-alive',  # 表示是否需要持久连接
		}
		r = requests.get(url, headers=HEADER)

		# print(r.content.decode("utf-8"))
		# import json
		soup = BeautifulSoup(r.content.decode("utf-8"), "html.parser")
		items = soup.find("table", {"class": "table table-bordered table-striped"}).find("tbody").find_all("tr")

		for item in items:
			ip = item.find("td", {"data-title": "IP"}).text
			port = item.find("td", {"data-title": "PORT"}).text

			print(ip, port)
			proxy_list.add("{}:{}".format(ip, port))

		time.sleep(5)
	return list(proxy_list)

def proxy_v2():
	proxy_list = []
	url = "https://proxy.seofangfa.com/"
	HEADER = {
		'User-Agent': random.choice(user_agent),  # 浏览器头部
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
		'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
		'Connection': 'keep-alive',  # 表示是否需要持久连接
	}
	r = requests.get(url, headers=HEADER)
	soup = BeautifulSoup(r.content.decode("utf-8"), "html.parser")
	items = soup.find("table", {"class": "table"}).find("tbody").find_all("tr")
	for item in items:
		columns = item.find_all("td")
		ip = columns[0].text
		port = columns[1].text

		print(ip, port)
		proxy_list.append("{}:{}".format(ip, port))
	return proxy_list

def proxy_v3():
	proxy_list = []
	url = "https://www.89ip.cn/index.html"
	HEADER = {
		'User-Agent': random.choice(user_agent),  # 浏览器头部
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
		'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
		'Connection': 'keep-alive',  # 表示是否需要持久连接
	}
	r = requests.get(url, headers=HEADER)
	soup = BeautifulSoup(r.content.decode("utf-8"), "html.parser")
	items = soup.find("table", {"class": "layui-table"}).find("tbody").find_all("tr")

	for item in items:
		columns = item.find_all("td")
		ip = columns[0].text.strip()
		port = columns[1].text.strip()

		print(ip, port)
		proxy_list.append("{}:{}".format(ip, port))
	return proxy_list


def proxy_v4():
	proxy_list = []
	url = "https://ip.ihuan.me/"
	HEADER = {
		'User-Agent': random.choice(user_agent),  # 浏览器头部
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
		'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
		'Connection': 'keep-alive',  # 表示是否需要持久连接
	}
	r = requests.get(url, headers=HEADER)
	content = r.content.decode("utf-8")
	# print(content)
	soup = BeautifulSoup(content, "html.parser")
	items = soup.find("table", {"class": "table table-hover table-bordered"}).find("tbody").find_all("tr")
	#
	for item in items:
		columns = item.find_all("td")
		ip = columns[0].text.strip()
		port = columns[1].text.strip()
	#
		print(ip, port)
		proxy_list.append("{}:{}".format(ip, port))
	return proxy_list

def proxy_v5():
	proxy_list = []
	url = "http://ip.yqie.com/ipproxy.htm"
	HEADER = {
		'User-Agent': random.choice(user_agent),  # 浏览器头部
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
		'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
		'Connection': 'keep-alive',  # 表示是否需要持久连接
	}
	r = requests.get(url, headers=HEADER)
	content = r.content.decode("utf-8")
	# print(content)
	soup = BeautifulSoup(content, "html.parser")
	items = soup.find("table", {"id": "GridViewOrder"}).find_all("tr")
	# #
	for item in items:
		columns = item.find_all("td")
		if len(columns) == 0:
			continue
		ip = columns[0].text.strip()
		port = columns[1].text.strip()
	#
		print(ip, port)
		proxy_list.append("{}:{}".format(ip, port))
	return proxy_list
	# print(r.content.decode("utf-8"))
def proxy_v6():
	proxy_list = []
	url = "https://ip.jiangxianli.com/?page=1"
	HEADER = {
		'User-Agent': random.choice(user_agent),  # 浏览器头部
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
		'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
		'Connection': 'keep-alive',  # 表示是否需要持久连接
	}
	r = requests.get(url, headers=HEADER)
	content = r.content.decode("utf-8")
	print(content)
	soup = BeautifulSoup(content, "html.parser")
	items = soup.find("table", {"id": "GridViewOrder"}).find_all("tr")
	# # #
	# for item in items:
	# 	columns = item.find_all("td")
	# 	if len(columns) == 0:
	# 		continue
	# 	ip = columns[0].text.strip()
	# 	port = columns[1].text.strip()
	# #
	# 	print(ip, port)
	# 	proxy_list.append("{}:{}".format(ip, port))
	return proxy_list


def get_proxy():
	d = {0: proxy_v1,
		 1: proxy_v2,
		 2: proxy_v3,
		 3: proxy_v4,
		 4: proxy_v5}
	rdi = random.randint(0, 4)

	proxy_list = d[rdi]()
	# print(len(proxy_list))
	return proxy_list


if __name__ == "__main__":
	proxy_v4()


