#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/6/11 18:20
    @Author  : jack.li
    @Site    : 
    @File    : investor_kg.py

"""
import os
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

engine = create_engine('sqlite:///D:\\xxxx\\foo.db')

# engine是2.2中创建的连接
Session = sessionmaker(bind=engine)

# 创建Session类实例
session = Session()


class Investor(Base):
    # 指定本类映射到users表
    __tablename__ = 'investor'
    # 如果有多个类指向同一张表，那么在后边的类需要把extend_existing设为True，表示在已有列基础上进行扩展
    # 或者换句话说，sqlalchemy允许类是表的字集
    # __table_args__ = {'extend_existing': True}
    # 如果表在同一个数据库服务（datebase）的不同数据库中（schema），可使用schema参数进一步指定数据库
    # __table_args__ = {'schema': 'test_database'}

    # 各变量名一定要与表的各字段名一样，因为相同的名字是他们之间的唯一关联关系
    # 从语法上说，各变量类型和表的类型可以不完全一致，如表字段是String(64)，但我就定义成String(32)
    # 但为了避免造成不必要的错误，变量的类型和其对应的表的字段的类型还是要相一致
    # sqlalchemy强制要求必须要有主键字段不然会报错，如果要映射一张已存在且没有主键的表，那么可行的做法是将所有字段都设为primary_key=True
    # 不要看随便将一个非主键字段设为primary_key，然后似乎就没报错就能使用了，sqlalchemy在接收到查询结果后还会自己根据主键进行一次去重
    # 指定id映射到id字段; id字段为整型，为主键，自动增长（其实整型主键默认就自动增长）
    id = Column(Integer, primary_key=True, autoincrement=True)
    # 指定name映射到name字段; name字段为字符串类形，
    full_name = Column(String(100))
    short_name = Column(String(100))
    location = Column(String(200))

def table():




    Base.metadata.create_all(engine, checkfirst=True)


def investor_v1():

    path = "G:\\download2\\invest_data\\"

    for file in os.listdir(path):
        file_path = path+file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        # print(html)

        soup = BeautifulSoup(html, "html.parser")
        span = soup.find("div", {"class": "tr-idg clearfix"})
        full_name = span.find("h1").text.strip()

        info_list = span.find_all("p")
        print(info_list[0].text)
        print(info_list[1].text)
        print(info_list[2].text.replace("\n", "").strip())

        # obj = Investor(full_name=title, short_name=short_name, location=location)
        # session.add(obj)
        # session.commit()

        # print("====================")

def investor_v2():
    path = "G:\download2\invest_data2\\"

    for file in os.listdir(path):
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        # print(html)

        soup = BeautifulSoup(html, "html.parser")
        span = soup.find("div", {"class": "info"})
        title = list(span.find("h1").strings)
        infos = span.find_all("li")
        print(title)
        print(infos[1].text)
        # break

def investor_v3():
    path = "G:\download2\invest_data4_plus\\"

    for file in os.listdir(path):
        file_path = path + file
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        # print(html)

        soup = BeautifulSoup(html, "html.parser")
        span = soup.find("div", {"class": "left_list"})
        # print(span)
        if span is None:
            continue
        short_name = soup.find("div", {"class": "head_tit"}).text.strip()
        infos = span.find_all("li")
        title = infos[0].find("span").text

        location = infos[3].find("span").text

        obj = Investor(full_name=title, short_name=short_name, location=location)
        session.add(obj)
        session.commit()

        print(title, location, short_name)
        # title = list(span.find("h1").strings)
        # infos = span.find_all("li")
        # print(title)
        # print(infos[1].text)
        # break


def tyc_investor():
    soup = BeautifulSoup(html, "html.parser")
    zz_list = []

    item_list = soup.find_all("a", {"class": "brand sv-search-company-brand"})
    for item in item_list:
        span_label = item.find("span", {"class": "tag-common -primary-bg ml8"})
        if span_label and span_label.text == "投资机构":
            print(row["investor"])
            company = item.find("span", {"class": "search-company-name hover"})
            if company:
                # bit += 1
                zz_list.append(company.text)

if __name__ == "__main__":
    investor_v1()
