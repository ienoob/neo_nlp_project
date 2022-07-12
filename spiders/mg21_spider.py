#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from spiders.base_spider import get_commom_content


url = "https://gs.amac.org.cn/amac-infodisc/res/pof/manager/101000000138.html"

content = get_commom_content(url)

print(content)
