#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/8 12:09
    @Author  : jack.li
    @Site    : 
    @File    : base_spider.py

"""
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

def get_content(url, retry=0, max_time=1):
    if retry >= max_time:
        print("retry out of times")
        return None
    HEADER = {
        'User-Agent': random.choice(user_agent),  # 浏览器头部
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
        'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
        'Connection': 'keep-alive',  # 表示是否需要持久连接
        "Cookie": """TYCID=ef3e4550fbda11eba924b976955f30df; ssuid=8642491674; _ga=GA1.2.227866782.1640593980; _bl_uid=tLkCXygFbbU6F6p3wh87nIRckpg1; aliyungf_tc=c88e09c402f03b1c0e6ea8edd5fdff7956a467caa608697bd8eb7c22644d1aa9; csrfToken=M8CAKSVuUD-MDC4KoSY9fnEp; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22285133987%22%2C%22first_id%22%3A%2217dfb0567b7322-07db8fcc64f1c2-978183a-2073600-17dfb0567b8105%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E8%87%AA%E7%84%B6%E6%90%9C%E7%B4%A2%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC%22%2C%22%24latest_referrer%22%3A%22https%3A%2F%2Fwww.google.com%2F%22%7D%2C%22%24device_id%22%3A%2217dfb0567b7322-07db8fcc64f1c2-978183a-2073600-17dfb0567b8105%22%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTgxM2Q4MGVjOTgxYjUtMGFmYTc4MTk0OGIwZmQtOTc4MTgzYS0yMDczNjAwLTE4MTNkODBlYzk5OTU0IiwiJGlkZW50aXR5X2xvZ2luX2lkIjoiMjg1MTMzOTg3In0%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%22285133987%22%7D%7D; bannerFlag=true; tyc-user-info=%7B%22state%22%3A%220%22%2C%22vipManager%22%3A%220%22%2C%22mobile%22%3A%2218516636317%22%7D; tyc-user-info-save-time=1654594297736; auth_token=eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIxODUxNjYzNjMxNyIsImlhdCI6MTY1NDU5NDI5NiwiZXhwIjoxNjU3MTg2Mjk2fQ.fdd7SW_GcqwRtSjw4Usi97RmH4F5G2z88B4kTS1-EDa01TXeqVZxPy17FK0ZycLbH1UTxTH4zEovIEkYOv49mA; Hm_lvt_e92c8d65d92d534b0fc290df538b4758=1654594387; creditGuide=1; RTYCID=eed8fab4b63d435b8b50e26dab4752cd; _gid=GA1.2.388906835.1654594389; acw_tc=b65cfd5416545978526732425e350c52ee5340b4d9a8305866b1a06adc7dde; acw_sc__v2=629f28dc5dfa9a1aaede851157e8c8ed3f429ff7; Hm_lpvt_e92c8d65d92d534b0fc290df538b4758=1654597862; cloud_token=62e120e1bb0c4b3dadf434b0f71e843b; cloud_utm=aed7150ac4724d8ea0c3e6caf73f312c; searchSessionId=1654597873.35882933""",


    }

    try:
        r = requests.get(url, headers=HEADER)
    except Exception as e:
        print("retry {}".format(retry))
        return get_content(url, retry + 1)


    return r.content.decode("utf-8")





def get_commom_content(url, retry=0, max_time=3, use_proxy=False, cookie=""):
	if retry >= max_time:
		print("retry out of times")
		return None
	HEADER = {
		'User-Agent': random.choice(user_agent),  # 浏览器头部
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
		'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
		'Connection': 'keep-alive',  # 表示是否需要持久连接

	}
	if cookie:
		HEADER["cookie"] = cookie

	try:
		r = requests.get(url, headers=HEADER)
	except Exception as e:
		print("retry {}".format(retry))
		return get_commom_content(url, retry + 1)
	if r.status_code == 403:
		print("403 warning")
		return None
	try:
		content = r.content.decode("utf-8")
	except Exception as e:
		print(e)
		content = None
	return content


class BaseSpider(object):
	pass
