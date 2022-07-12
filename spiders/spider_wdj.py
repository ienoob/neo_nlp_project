#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

from spiders.base_spider import get_commom_content
from spiders.selenium_spider import get_wdj_cookie

# def spider():
#
#     for i in range(1, 100000):
# cookie = get_wdj_cookie()
url = "https://www.wandoujia.com/category/5029"

content = get_commom_content(url)

print(content)


