#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import pandas as pd
from bs4 import BeautifulSoup

short_name_list = []

with open("D:\data\self-data\\finance_company_alias.txt", "w", encoding="utf-8") as f:
    f.write("")

def data_v1():
    data_path = "D:\Work\git\\neo_nlp_project\spiders\invest_data\\"

    data_list = os.listdir(data_path)

    # soup = BeautifulSoup("lxml")
    iv = 0

    for file in data_list:
        file_path = data_path + file

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        soup = BeautifulSoup(data, "lxml")

        span = soup.find(name="div", attrs={"class": "tr-idg clearfix"})
        title = list(span.find("h1").strings)

        # print(title[0])
        short_name = list(span.find("p").strings)
        if len(short_name) < 2 or len(title)<1:
            continue
        iv += 1
        # print(short_name[1], title[0], "1")

        # short_name_list.append({"full_name": title[0], "short_name": short_name[1]})
        # short_name_list.append((title[0], short_name[1]))
        print(len(short_name_list), iv)

        with open("D:\data\self-data\\finance_company_alias.txt", "a+", encoding="utf-8") as f:
            f.write("{}\t{}\t{}\n".format(title[0], short_name[1], "v1"))


def data_v2():
    data_path = "D:\Work\git\\neo_nlp_project\spiders\invest_data2\\"

    data_list = os.listdir(data_path)
    iv = 0
    for file in data_list:
        file_path = data_path + file

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        soup = BeautifulSoup(data, "lxml")

        span = soup.find("div", attrs={"class": "info"})
        title = list(span.find("h1").strings)
        if len(title)>1:
            print(title[0], "====", title[1])

            # short_name_list.append({"full_name": title[0], "short_name": title[1]})

            print(len(short_name_list), iv)
            iv += 1

            with open("D:\data\self-data\\finance_company_alias.txt", "a+", encoding="utf-8") as f:
                f.write("{}\t{}\t{}\n".format(title[0], title[1], "v2"))


def data_v3():
    data_path = "D:\Work\git\\neo_nlp_project\spiders\invest_data3\\"

    data_list = os.listdir(data_path)

    for file in data_list:
        file_path = data_path + file

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        soup = BeautifulSoup(data, "lxml")


def data_v4():
    data_path = "D:\Work\git\\neo_nlp_project\spiders\invest_data4_plus\\"

    data_list = os.listdir(data_path)
    iv = 0
    for file in data_list:
        file_path = data_path + file

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        soup = BeautifulSoup(data, "lxml")
        title = soup.find("div", attrs={"class": "head_tit"})
        if not title:
            continue
        short_name = title.text.strip()
        print(short_name, title, "v4")

        name = soup.find("ul", attrs={"class": "left_listbasic clearfix"}).find("li").find("span").text.strip()
        # short_name_list.append({"full_name": name, "short_name": short_name})
        print(len(short_name_list), iv)
        iv += 1

        with open("D:\data\self-data\\finance_company_alias.txt", "a+", encoding="utf-8") as f:
            f.write("{}\t{}\t{}\n".format(name, short_name, "v4"))

        # break
data_v1()
data_v2()
data_v4()

# df = pd.DataFrame(short_name_list)
# print(df.shape)
# df.to_csv("short_name.csv", encoding="utf-8", index=False)


