#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/5/27 23:20
    @Author  : jack.li
    @Site    : 
    @File    : qcc_search.py

"""
from spiders.base_spider import get_commom_content

url = "https://www.qcc.com/web/search?key=%E9%87%91%E6%88%88%E6%88%88%E9%A6%99%E6%B8%AF%E8%B1%89%E6%B2%B9%E9%B8%A1"

content = get_commom_content(url)

print(content)