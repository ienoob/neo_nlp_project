#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from spiders.base_spider import get_content_v2


url = "https://www.qixin.com/search?key=%E8%A5%BF%E4%BA%9A&page=1"
content = get_content_v2(url)

print(content)
