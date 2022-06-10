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
from spiders.base_spider import get_commom_content
from spiders.selenium_spider import get_crunchbase_cookie

data_path = "D:\\xxxx\\investee.json"
data_path = "D:\\xxxx\\po_entity.json"
data_path = "D:\\xxxx\\investee_entity.json"
data_path = "D:\\xxxx\\public_opinion_company_label.json"
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
import hashlib
cookie = get_crunchbase_cookie()
for k, v in data_dict_list:
    if fv(k):
        print(k, v)
        urlcode = parse.urlencode({"query": k})
        url = "https://www.crunchbase.com/v4/data/autocompletes?{}&collection_ids=organizations&limit=25&source=topSearch".format(urlcode)
        print(url)

        file_md5_indx = hashlib.md5(k.encode()).hexdigest()
        file_path = "F:\\download2\\crunchbase\\{}.json".format(file_md5_indx)
        if os.path.exists(file_path):
            continue

        content = get_commom_content(url, cookie=cookie)
        if content == "403":
            print("page 403 warning")
            # print(content)
            continue
        if content is None:
            continue

        if content[0] != "{":
            break
            # print(content[:5])
            # # time.sleep(20)
            # continue
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
