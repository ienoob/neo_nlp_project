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
        "Cookie": """TYCID=dae1530066f611ebb18b930297228e53; ssuid=871731868; _ga=GA1.2.929265637.1612449602; _bl_uid=h3l8L0n9t53gjwp1bsOtkvI9tvz2; hkGuide=1; _gid=GA1.2.1900400168.1652339881; creditGuide=1; jsid=SEO-BING-ALL-SY-000001; RTYCID=2ccdc25480be4db2bf4697390e3aea84; aliyungf_tc=85660ff1c8a531db2b3c1b9a701e8a9c941bc680b9282196d5fad5e35e49d91c; csrfToken=82FfioWygmm-bxByBBs_qQK_; bannerFlag=true; Hm_lvt_e92c8d65d92d534b0fc290df538b4758=1652757213,1652767415,1652775741,1652785121; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22285133987%22%2C%22first_id%22%3A%221776d7c54936fe-064c284e644fc-50391c46-1327104-1776d7c54947af%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E8%87%AA%E7%84%B6%E6%90%9C%E7%B4%A2%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC%22%2C%22%24latest_referrer%22%3A%22https%3A%2F%2Fcn.bing.com%2F%22%7D%2C%22%24device_id%22%3A%221776d7c54936fe-064c284e644fc-50391c46-1327104-1776d7c54947af%22%7D; acw_tc=0a324fa816527933324308701e4fb49c25578bd11a3543d5eb16e771bb5048; searchSessionId=1652794501.16534737; cloud_token=177c019a4d304b989132503a10b53c3e; cloud_utm=0de12aaddf20409298f6ebb7db4a628b; relatedHumanSearchGraphId=2329375836; relatedHumanSearchGraphId.sig=ZeEaDkkj43CB0TjLn0gRj6QTh2BDOWaS6VCBf3w8JmE; _gat_gtag_UA_123487620_1=1; tyc-user-info={%22state%22:%220%22%2C%22vipManager%22:%220%22%2C%22mobile%22:%2218516636317%22}; tyc-user-info-save-time=1652794615720; auth_token=eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIxODUxNjYzNjMxNyIsImlhdCI6MTY1Mjc5NDYxNiwiZXhwIjoxNjU1Mzg2NjE2fQ.ObyK2D_xhweLb9BW4iXRUNmsBtdTMmIa8b4V5KLxuIFGF81azEBJvVYw-fM-bIfRqTCrJnJOxlEjYEIqM73YlA; tyc-user-phone=%255B%252218516636317%2522%255D; Hm_lpvt_e92c8d65d92d534b0fc290df538b4758=1652794619; _dd_s=logs=1&id=1356b318-70fb-4421-a225-7194e4c64c9c&created=1652793330141&expire=1652795531047"""

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