#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/27 20:55
    @Author  : jack.li
    @Site    : 
    @File    : crunchbase_search.py

"""
import time
import json
from urllib import parse
from spiders.base_spider import get_commom_content, user_agent
from spiders.get_proxy import get_proxy

data_path = "D:\\xxxx\\investee.json"
data_path = "D:\\xxxx\\po_entity.json"
data_path = "D:\\xxxx\\investee_entity.json"
# data_path = "D:\\xxxx\\investor_entity.json"
with open(data_path, "r") as f:
    data_json = f.read()
data_dict = json.loads(data_json)
data_dict_list = [(k, v) for k, v in data_dict.items()]
data_dict_list.sort(key=lambda x: x[1], reverse=True)

def  is_alphabet(uchar):

    if  ('\u0041'  <=  uchar<='\u005a')  or  ('\u0061'  <=  uchar<='\u007a'):

        return  True

    else:

        return  False
def fv(input_x):
    # print(type(input_x))
    state = 0
    for x in input_x:
        # print(x, x.isdigit(), x.isalpha())
        if x in [" "]:
            continue
        if x.isdigit():
            continue
        if is_alphabet(x):
            continue
        return False
    return True
import os
import time
import hashlib
import random
import requests

proxy_list = get_proxy()
def get_data(url, retry=0, max_time=30):
    if retry>=max_time:
        print("retry out of times")
        return None
    HEADER = {
        'User-Agent': random.choice(user_agent),  # 浏览器头部
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',  # 客户端能够接收的内容类型
        'Accept-Language': 'en-US,en;q=0.5',  # 浏览器可接受的语言
        'Connection': 'keep-alive',  # 表示是否需要持久连接
    }
    if len(proxy_list) == 0:
        time.sleep(3)
        p_list = get_proxy()
        for p in p_list:
            proxy_list.append(p)
        if len(proxy_list) == 0:
            print("proxy num empty")
            return None
    ip_proxy = proxy_list.pop(0)
    proxies = {
        'http': ip_proxy,
        'https': ip_proxy,
    }
    try:
        r = requests.get(url, headers=HEADER, proxies=proxies)
    except Exception as e:
        print("retry {}".format(retry))
        return get_data(url, retry+1)
    proxy_list.append(ip_proxy)

    return r.content.decode("utf-8")

for k, v in data_dict_list:
    if fv(k):
        print(k, v)
        urlcode = parse.urlencode({"query": k})
        url = "https://www.crunchbase.com/v4/data/autocompletes?{}&collection_ids=organizations&limit=25&source=topSearch".format(urlcode)
        print(url)

        file_md5_indx = hashlib.md5(k.encode()).hexdigest()
        file_path = "G:\\download2\\crunchbase\\{}.json".format(file_md5_indx)
        if os.path.exists(file_path):
            continue
        cookie = 'cid=Cig0BGKFAEtfYAAugcq2Ag==; _pxvid=7a9b2d0f-d6b5-11ec-ad7c-686c4d627659; _ga=GA1.2.455298364.1652883917; _hp2_props.973801186=%7B%22Logged%20In%22%3Afalse%2C%22Pro%22%3Afalse%2C%22cbPro%22%3Afalse%7D; drift_aid=00a4d3fd-b59c-47ab-bc3d-af0ee44171f2; driftt_aid=00a4d3fd-b59c-47ab-bc3d-af0ee44171f2; _gcl_au=1.1.1452636244.1652885814; _biz_uid=a3c84577b10c4432ed5c2a931e37fbc7; __qca=P0-1170339832-1652885840794; _mkto_trk=id:976-JJA-800&token:_mch-crunchbase.com-1652885842189-63529; _hjSessionUser_977177=eyJpZCI6IjU0ZmI3ZmQ5LThkYzgtNWExYy1iN2ZkLTdiNjE0ZWVjOWQ4MyIsImNyZWF0ZWQiOjE2NTI4ODU4MzE1NDcsImV4aXN0aW5nIjp0cnVlfQ==; __cflb=0pg1SSbFg8JtpNt3MarxMvaKq17VG3oT626gafhe; _gid=GA1.2.826794711.1653652973; _biz_nA=3; _biz_flagsA=%7B%22Version%22%3A1%2C%22ViewThrough%22%3A%221%22%2C%22Mkto%22%3A%221%22%2C%22XDomain%22%3A%221%22%7D; _biz_pendingA=%5B%5D; xsrf_token=1rQJsP/ymsoiYNKYfMmgbXMy8LAGVRUjZ6YKhlqnVvU; pxcts=df1245e8-de33-11ec-aedb-424172446d65; _pxhd=LfsiypOHPim31Htn-Bnf7dv9AhryYvISs9Qe0YJsDBWr-7ohNvHxlQu4KBOKabBLDTMFOPdelZZL5i9NNBiBSw==:oypTAewh1bYTJkZ6yGg1k7xgU7BlCFCtI1tWp9TBn9p0wnuxG/YEQCaOyB8ARXkF8VJenjzGhJ9wBKAbz8cVA5930y7ctJZthX6BORZwv/Q=; OptanonConsent=isIABGlobal=false&datestamp=Sat+May+28+2022+16%3A37%3A41+GMT%2B0800+(GMT%2B08%3A00)&version=6.23.0&hosts=&consentId=7ec76659-7f36-4778-b66d-fe2d5e0e5b75&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CBG7%3A1%2CC0004%3A1%2CC0002%3A1&AwaitingReconsent=false&geolocation=CN%3BSH; OptanonAlertBoxClosed=2022-05-28T08:37:41.495Z; _hp2_id.973801186=%7B%22userId%22%3A%227471882867854515%22%2C%22pageviewId%22%3A%222871741810684237%22%2C%22sessionId%22%3A%226738260525800898%22%2C%22identity%22%3Anull%2C%22trackerVersion%22%3A%224.0%22%7D; _px3=189184170e7f469bec0196e2cebc05dee3a6772cb89781a4e93a07d6537b24f8:OVq2ujFsWj/8LiHW+GaslG7MKxWLActzhYOp2wcypingasBCfrtGMlvU0cT7iu3SJfchQpsPlQQkRncRwGkpOg==:1000:Om3zpSOptyAtV0LeS+nv0aAe58q8D54u3xIV52P/jkQaizywG2Jm+rsn3TqspozEEQO/P/8px0t83ce3Q9zVZQAgve7a13guXcvkKJvDLhiF8Myt8GnYDJuRhzVUXO3zhlIYHZYww+8e4jWbyn1Qy0eAQTHBdXYGxGPfvpOQjzZxI45cONE5q09pLz5J42FkfsL8Bo0NdJP0sa9jGErGJg=='
        content = get_data(url)
        if content == "403":
            print("page 403 warning")
            # print(content)
            continue
        if content is None:
            continue

        if content[0] != "{":
            print(content[:5])
            # time.sleep(20)
            continue
        # try:
        #     d = json.loads(content)
        # except Exception as e:
        #     continue

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # time.sleep(30)
        print("{} complete!".format(k))



#

# print(content)
# print(f("香港查氏家族基金"))