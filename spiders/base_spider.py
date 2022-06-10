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


def get_content(url, retry=0, max_time=1, cookies=dict()):
    if retry >= max_time:
        print("retry out of times")
        return None
    HEADER = {
        'User-Agent': random.choice(user_agent),  # 浏览器头部
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
        'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
        'Connection': 'keep-alive',  # 表示是否需要持久连接
        # "Cookie": """TYCID=ef3e4550fbda11eba924b976955f30df; ssuid=8642491674; _ga=GA1.2.227866782.1640593980; _bl_uid=tLkCXygFbbU6F6p3wh87nIRckpg1; tyc-user-info=%7B%22state%22%3A%220%22%2C%22vipManager%22%3A%220%22%2C%22mobile%22%3A%2218516636317%22%7D; tyc-user-info-save-time=1654594297736; auth_token=eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIxODUxNjYzNjMxNyIsImlhdCI6MTY1NDU5NDI5NiwiZXhwIjoxNjU3MTg2Mjk2fQ.fdd7SW_GcqwRtSjw4Usi97RmH4F5G2z88B4kTS1-EDa01TXeqVZxPy17FK0ZycLbH1UTxTH4zEovIEkYOv49mA; Hm_lvt_e92c8d65d92d534b0fc290df538b4758=1654594387; creditGuide=1; RTYCID=eed8fab4b63d435b8b50e26dab4752cd; searchSessionId=1654597873.35882933; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22285133987%22%2C%22first_id%22%3A%2217dfb0567b7322-07db8fcc64f1c2-978183a-2073600-17dfb0567b8105%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22%24device_id%22%3A%2217dfb0567b7322-07db8fcc64f1c2-978183a-2073600-17dfb0567b8105%22%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTgxM2Q4MGVjOTgxYjUtMGFmYTc4MTk0OGIwZmQtOTc4MTgzYS0yMDczNjAwLTE4MTNkODBlYzk5OTU0IiwiJGlkZW50aXR5X2xvZ2luX2lkIjoiMjg1MTMzOTg3In0%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%22285133987%22%7D%7D; aliyungf_tc=201c6f78a095caa2e4b5cc982b6453fdf259df7e66d640953f4be0c08b1cf723; csrfToken=Xv66m0-mQgs2xkYP_wTD-hBK; bannerFlag=true; acw_tc=b65cfd5a16547458762278800e038635e852647a5414818c70e2e268c61111; acw_sc__v2=62a16b144b8fad1c5c2e327949eb6a0514a8ecbf""",


    }

    try:
        r = requests.get(url, headers=HEADER, cookies=cookies)
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


def get_content_v2(url, retry=0, max_time=3):
	if retry >= max_time:
		print("retry out of times")
		return None
	HEADER = {
		":authority": "www.qixin.com",
":method": "GET",
":path": "/search?key=%E8%A5%BF%E4%BA%9A&page=1",
":scheme": "https",
"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
"accept-encoding": "gzip, deflate, br",
"accept-language": "zh,zh-TW;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6",
"cache-control": "max-age=0",
"cookie": "fid=d7578757b70ea35cbbe7f1cc514cfb46; flogger_fpid=d7578757b70ea35cbbe7f1cc514cfb46; pdid=s%3AH2Hw2loZe7Noe9TDGip3A4UxhCGaNnU-.ThvqPNwvznSLt7bO91B08WZEvgk4CsX2eM1BtCu6ers; Hm_lvt_52d64b8d3f6d42a2e416d59635df3f71=1654828926; acw_tc=76b20fec16548509463632682e0f6a4893514ca5d0b69c0ac10b9de088d011; Hm_lpvt_52d64b8d3f6d42a2e416d59635df3f71=1654851106; ssxmod_itna=YqUOY5AK7IODCFDXDnD+r8x9mAXmpPDQQrbqd+ODlOpYxA5D8bD6BEtYtmY/ijfyed5ncRvAa5jDlGIqLbfbweDHxY=DUciRqoD44GTDt4DTD34DYDixibOxi5GRD0KDFbVztZcDm4i3ExDbDiW8wxGCDeKD0ouaDQKDupKP8Y+YErBaY35m06DhhmKD9ooDsp0EORYff34zeK8YcAqDUieUo7L5dm0sLQKDXoQDvrvCgpPm9RIzcZyoWjRDWYrtzpidd72q5ihYdAhcq02BeYRy7QRqmBRKemheqpxDi=4KfjGDD===; ssxmod_itna2=YqUOY5AK7IODCFDXDnD+r8x9mAXmpPDQQrbqd+D6EfO=D0vaRqP03q+988ed0SDnRDkFYDnFid9zYbI=1554P5P73EzB3r1FjKlFH=HqHWSwmd=bto2=HymQxtEXbU0F1=oXZzBwq7SK=eOGrGEXZzxIPbnK6Roid0kKKk+fcADN3lBrYFYhLCOXxnjA5GTmDHib6WbelGAA2W4=3wroyzeUgCdnGiT=8vLdEBRjLIkfC7qQCCROv1DUn3SHlSR+L4CfPLPT8=RI/fqXgFuOPEPNPLwom2bZL0wDBau6wngllyOFmSakdgFYhblqsYr=Kns1gWY2=nbHAuWxgDfOKiDo7oewrYMwrEhbAfz3EzL2dlrQQ8ox25Yk+5Bh0BxLCKoeKUY=+Am5haRrbwgPBrooibA+5ogDL3KwY=6MwDqO4r7WLd+D8MEkbOmmM3A2O/OL0MKxgCqdvrpNPrkxrqxf7d8iM3UwOtpl5h0QYkDuOxMWKm3pQ=tLONtpYC1oo+QeANS7D87xCm3soxwQtVG6eroMxaGxLFD=UfvktNuuRxbn8t5pFoCqXUSmX+o0ruQZxfDDwgeDjKDeuo+0Yatxtoe=iDD=",
"referer": "https://www.qixin.com/",
"sec-ch-ua": '"Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
"sec-ch-ua-mobile": "?0",
"sec-ch-ua-platform":  "Windows",
"sec-fetch-dest":  "document",
"sec-fetch-mode":  "navigate",
"sec-fetch-site":  "same-origin",
"sec-fetch-user":  "?1",
"upgrade-insecure-requests":  1,
"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"

	}

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
