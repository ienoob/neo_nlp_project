#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import re
import os
import json
import hashlib
import pandas as pd
from pytorch.other_info_extraction.policy_information_extractor_v3 import Document

cn2en_dict = {
                "项目名称": "project_name",
                "项目金额/补贴金额": "project_profit",
                "企业注册地区": "company_register_area",
                "企业注册日期": "company_register_date",
                "企业成立日期": "company_register_date",
                "注册资本": "company_register_capital",
                "企业性质": "company_nature",
                "企业类型": "company_type",
                "企业技术能力": "company_tech_ability",
                "企业资质": "company_qualification",
                "人才资质": "person_qualification",
                "企业行业": "company_industry",
                "所属领域": "company_industry",
                "企业人数": "company_population",
                "领军人数": "leader_num",
                "大专及以上人数": "college_students_num_above",
                "本科及以上人数": "bachelor_degree_num_above",
                "硕士及以上人数": "master_degree_num_above",
                "博士及以上人数": "doctor_degree_num_above",
                "企业总资产": "company_total_assets",
                "企业实收资本": "company_total_assets",
                "净资产总额": "company_total_assets",
                "净利润": "net_profit",
                "营业收入": "operating_income",
                "税收": "tax",
                "知识产权总数": "intellectual_property_rights_num",
                "专利总数": "patent_num",
                "发明专利": "patent_num",
                "注册商标总数": "registered_trademarks_num",
                "软件著作权总数": "software_copyright_num",
                "软件著作权": "software_copyright_num",
                "申报材料": "application_materials",
                "申报流程": "report_processes",
                "研究开发投入": "a1",
                "研发投入占营业收入比例": "a3",
                "研发经费支出占主营业务收入": "a4",
                "无失信行为": "a5",
                "研发人数": "a6",
                "技术中心工作人员": "a7",
                "三年营业收入平均增长率": "a8",
                "平均营业利润率": "a9",
                "研究开发费用占销售收入的比重": "a10",
                "软件产品评估": "a11",
                "业务收入": "a12",
                "研发费用总额占同期销售收入总额的比例": "a13",
                "运营时间": "a14",
                "企业注册年限": "a14",
                "电子商务年交易额": "a15",
                "电子商务服务业年营业收入": "a16",
                "最后一轮投后估值": "a17",
                "最新一轮投后估值": "a18",
                "上年度销售收入": "a19",
                "三年销售收入增长率": "a20",
                "净利润平均增长率": "a21",
                "2020年度相关业务收入": "a22",
                "项目注册资本": "a23",
                "购置设备的发票金额": "a24",
                "境外母公司资产总额": "a25",
                "境外母公司持股比例": "a26",
                "投资的境内外独立法人企业": "a27",
                "设立的境内外分支机构": "a28",
                "授权管理（服务）的境内外独立法人企业": "a29",
                "授权管理（服务）的境内外分支机构": "a30"
            }

logic_cn2en_dict = {
    "大于": ">",
    "小于": "<",
    "大于等于": ">=",
    "小于等于": "<=",
    "等于": "==",
    "不等于": "!=",
    "属于": "∈"
}

class Dataset(object):
    def __init__(self, path, way="list", param=34):
        self.path = path
        self.way = way
        self.param = param

    def __iter__(self):
        if self.way == "list-txt":
            for file in os.listdir(self.path):
                file_path = self.path + "\\" + file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
                yield {"title": file.split(".")[0], "text": data.strip()}
        elif self.way == "jsonline":
            with open(self.path, "r", encoding="utf-8") as f:
                data = f.read()
            for dt in data.split("\n"):
                if dt.strip() == "":
                    continue
                # print(dt)
                dt_content = json.loads(dt)
                yield dt_content["content"]
        elif self.way == "list-json":
            for file in os.listdir(self.path):
                file_name = file.split(".")[0]
                file_path = self.path + "\\" + file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = f.read()
                content = json.loads(data)["content"].strip()
                yield {"text": content, "title": file_name}
        elif self.way == "xlsx":
            df = pd.read_excel(self.path, sheet_name="政策知识抽取")
            for title in df.columns.values:
                if title in ['分类', '指标名称', "政策指标"]:
                    continue
                if isinstance(df[title][self.param], float):
                    continue
                data_json = json.loads(df[title][self.param])

                yield {"title": title, "text": data_json["content"].strip()}

        elif self.way == "list-txt-deep":
            for file in os.listdir(self.path):
                if file == "html":
                    continue
                file_path = self.path + "\\" + file
                if os.path.isdir(file_path):
                    for sub_file in os.listdir(file_path):
                        sub_file_path = file_path + "\\" + sub_file
                        with open(sub_file_path, "r", encoding="utf-8") as f:
                            data = f.read()
                        data = data.strip()

                        if len(data ) > 50:
                            yield {"title": "", "text": data}



def extract_info(doc, content):
    doc.parse_content(content)
    doc.display_document()
    res = doc.parse_half_struction()
    print(res)

    def f(parse_info):
        parse_res = {
            "others": []
        }
        for k, v in cn2en_dict.items():
            if re.match("a[0-9]{1,3}", v):
                continue
            if k in ["申报材料", "申报流程"]:
                parse_res[v] = []
            elif k in ["项目名称"]:
                parse_res[v] = ""
            elif k in ["项目金额/补贴金额"]:
                parse_res[v] = []
            else:
                parse_res[v] = {}
        ignore_list = ["链接资料"]
        for zhibiao_key in parse_info["zhibiao"]:
            if len(zhibiao_key) == 3:
                print(zhibiao_key)
            k, lg, v, c = zhibiao_key
            if k in ignore_list:
                continue
            if k not in cn2en_dict:
                parse_res["others"].append([k, logic_cn2en_dict[lg], v, c])
            else:
                en_key = cn2en_dict[k]
                if re.match("a[0-9]{1,3}", en_key):
                    parse_res["others"].append([k, logic_cn2en_dict[lg], v, c])
                else:
                    parse_res[en_key] = {
                        "value": v,
                        "logic": logic_cn2en_dict[lg],
                        "measure": c
                    }
        return parse_res

    project_infos = list()


    if len(res["project_infos"]) == 0:

        parse_info = doc.parse_precision(res["conditions"])
        parse_res = f(parse_info)

        parse_res["project_profit"] = res["benefit_infos"]
        parse_res["application_materials"] = res["text_materials"]
        parse_res["report_processes"] = res["report_processes"]
        project_infos.append(parse_res)

    else:
        for project_info in res["project_infos"]:
            print(project_info)
            parse_info = doc.parse_precision(project_info["conditions"])
            parse_res = f(parse_info)
            parse_res["project_name"] = project_info["project_name"]
            parse_res["application_materials"] = project_info.get("text_materials", [])
            parse_res["project_profit"] = project_info.get("benefit_infos", [])
            parse_res["report_processes"] = project_info.get("report_processes", [])

            project_infos.append(parse_res)
    doc.clear_data()
    return project_infos


if __name__ == "__main__":
    collect_data = []
    demo_path = "D:\data\\tianjin_dataset\政策信息抽取\\20211129" # list-json
    dataset = Dataset(demo_path, "list-json")
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    demo_path2 = "D:\data\\tianjin_dataset\政策信息抽取\chace_files\chace_files"
    dataset = Dataset(demo_path2, "list-txt-deep")
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    demo_path3 = "D:\data\\tianjin_dataset\政策信息抽取\\text"
    dataset = Dataset(demo_path3, "list-txt")
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    demo_path4 = "D:\data\\tianjin_dataset\政策信息抽取\政策content\政策content"
    dataset = Dataset(demo_path4, "list-json")
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    demo_path5 = "D:\data\\tianjin_dataset\政策信息抽取\政策原文\政策原文"
    dataset = Dataset(demo_path5, "list-json")
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))


    demo_path6 = "D:\data\\tianjin_dataset\政策信息抽取\政策匹配的企业要素-20211103.xlsx"
    dataset = Dataset(demo_path6, "xlsx")
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    demo_path7 = "D:\data\\tianjin_dataset\政策信息抽取\政策匹配的企业要素-20211126.xlsx"
    dataset = Dataset(demo_path7, "xlsx", 43)
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    demo_path8 = "D:\data\\tianjin_dataset\政策信息抽取\政策匹配的企业要素-20211203.xlsx"
    dataset = Dataset(demo_path8, "xlsx", 43)
    for data in dataset:
        collect_data.append(data)

    print(len(collect_data))

    filter_data = dict()
    for data in collect_data:
        data_id =  hashlib.md5(data["text"].encode()).hexdigest()
        if data_id not in filter_data:
            data["id"] = data_id
            filter_data[data_id] = data

    print(len(filter_data))

    df_data = [data for k, data in filter_data.items()]
    df = pd.DataFrame(df_data)

    df.to_csv("policy_data-20211228.csv", index=False)





    # demo_path = "Z:\政策抽取\政策原文解析.txt"
    # demo_path2 = "zc_json.jsonline"
    # demo_path3 = "Z:\政策抽取\政策文件.jsonline"
    # demo_path4 = "D:\data\\tianjin_dataset\政策信息抽取\\text"
    # path = "Z:\政策抽取\政策信息抽取\chace_files\chace_files\\3产业基金"
    # dataset = Dataset(demo_path4, "list")
    #
    # documents = []
    #
    # i = 0
    # limit = 0
    # doc = Document()
    # for data in dataset:
    #     i += 1
    #     if i < limit:
    #         continue
    #
    #     res = extract_info(doc, data)
    #     # print(res)
    #     if len(res):
    #         documents.append({"content": data, "event": res})
    #     print("=============================== {}".format(i))
    #
    # with open("policy.json", "w") as f:
    #     f.write(json.dumps(documents))



        # break

