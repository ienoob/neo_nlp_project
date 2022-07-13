#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/4/28 22:30
    @Author  : jack.li
    @Site    : 
    @File    : cnpedia_search.py

"""
import json
import requests
from urllib import parse


question = "打球的李娜和唱歌的李娜不是同一个人"
urlcode = parse.urlencode({"q": question})
url = "http://shuyantech.com/api/entitylinking/cutsegment?{}".format(urlcode)

# r = requests.get(url)
# print(json.dumps(json.loads(r.content), indent=4, ensure_ascii=False))

url2 = "http://kw.fudan.edu.cn/ddemos/abbr/?p=杭州优品互联网金融服务有限公司"
r = requests.get(url2)
print(json.dumps(json.loads(r.content), indent=4, ensure_ascii=False))