#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import re
from transformers import BertTokenizer, BertForQuestionAnswering, PreTrainedTokenizerFast

from tika import parser
from bs4 import BeautifulSoup

query_items = ["注册资本", "成立日期", "企业注册地区、地址", "企业类型", "企业性质", "企业资质", "企业人数", "科技人员数占比", "总资产", "销售收入", "服务收入占比"]
query_item_list = {
    "企业注册地区": ["企业注册地区、地址"],
    "企业资质": {
        "千百万工程重点企业": ["国家级", "省级", "市级", "区县级"],
        "100强企业": ["世界级", "国家级", "省级", "市级", "区县级"],
        "A级景区": ["AAAAA级旅游景区", "AAAA级旅游景区", "AAA级旅游景区", "AA级旅游景区", "A级旅游景区"],
        "CMA认证企业": ["国家级", "省级", "市级", "区县级"],
        "CMMI认证企业": ["CMMI-5", "CMMI-4", "CMMI-3", "CMMI-2", "CMMI-1"],
        "CNAS认证企业": ["国家级", "省级", "市级", "区县级"],
        "GLP认证企业": ["国家级", "省级", "市级", "区县级"],
        "ISCCC认证企业": ["一级", "二级", "三级", "四级"],
        "ISO认证企业": ["ISO14001", "ISO9001", "ISO27001", "ISO20000"],
        "ITSS认证企业": ["一级", "二级", "三级", "四级"],
        "版权重点企业": ["国家级", "省级", "市级", "区县级"],
        "成长型企业": ["国家级", "省级", "市级", "区县级"],
        "出口名牌企业": ["国家级", "省级", "市级", "区县级"],
        "创新型示范企业": ["国家级", "省级", "市级", "区县级"],
        "大数据应用示范企业": ["国家级", "省级", "市级", "区县级"],
        "瞪羚（培育）企业": ["国家级", "省级", "市级", "区县级"],
        "地标型企业": ["国家级", "省级", "市级", "区县级"],
        "独角兽（培育）企业": ["国家级", "省级", "市级", "区县级"],
        "高成长创新型企业（培育）": ["国家级", "省级", "市级", "区县级"],
        "高企培育入库企业": ["国家级", "省级", "市级", "区县级"],
        "高新技术企业": ["国家级", "省级", "市级", "区县级"],
        "规上企业": [],
        "技术创新示范企业": ["国家级", "省级", "市级", "区县级"],
        "技术先进型企业": ["国家级", "省级", "市级", "区县级"],
        "检验检测许可企业": ["国家级", "省级", "市级", "区县级"],
        "科技小巨人企业": ["国家级", "省级", "市级", "区县级"],
        "科技型中小企业": ["国家级", "省级", "市级", "区县级"],
        "老字号企业": ["国家级", "省级", "市级", "区县级"],
        "领军企业": ["国家级", "省级", "市级", "区县级"],
        "龙头骨干企业": ["国家级", "省级", "市级", "区县级"],
        "绿色商场": ["国家级", "省级", "市级", "区县级"],
        "民营科技企业": ["国家级", "省级", "市级", "区县级"],
        "农业科技型企业": ["国家级", "省级", "市级", "区县级"],
        "农业龙头企业": ["国家级", "省级", "市级", "区县级"],
        "平台经济重点企业": ["国家级", "省级", "市级", "区县级"],
        "企业上市培育入库企业": ["国家级", "省级", "市级", "区县级"],
        "软件企业": ["国家级", "省级", "市级", "区县级"],
        "商业老字号": ["国家级", "省级", "市级", "区县级"],
        "上市企业": ["海外上市", "主板上市", "中小板上市", "创业板上市", "新三板上市", "科创板上市"],
        "上云优秀企业": ["国家级", "省级", "市级", "区县级"],
        "示范/标杆企业": ["国家级", "省级", "市级", "区县级"],
        "示范村": ["国家级", "省级", "市级", "区县级"],
        "示范基地/园区": ["国家级", "省级", "市级", "区县级"],
        "示范街区/社区": ["国家级", "省级", "市级", "区县级"],
        "示范镇": ["国家级", "省级", "市级", "区县级"],
        "双创平台": ["国家级", "省级", "市级", "区县级"],
        "特色产业园区": ["国家级", "省级", "市级", "区县级"],
        "特色商业街（区）": ["国家级", "省级", "市级", "区县级"],
        "体系贯标企业": ["国家级", "省级", "市级", "区县级"],
        "先进型服务企业": ["国家级", "省级", "市级", "区县级"],
        "星级饭店": ["一星", "二星", "三星", "四星", "五星"],
        "“驰名商标”企业": ["国家级", "省级", "市级", "区县级"],
        "隐形冠军企业": ["国家级", "省级", "市级", "区县级"],
        "隐形小巨人企业": ["国家级", "省级", "市级", "区县级"],
        "优秀软件产品奖（“金慧奖”）": ["国家级", "省级", "市级", "区县级"],
        "知识产权贯标企业": [],
        "知识产权密集型企业": ["国家级", "省级", "市级", "区县级"],
        "知识产权示范企业": ["国家级", "省级", "市级", "区县级"],
        "知识产权优势企业": ["国家级", "省级", "市级", "区县级"],
        "中国服务业500强": ["国家级", "省级", "市级", "区县级"],
        "重点软件企业": ["国家级", "省级", "市级", "区县级"],
        "重点文化体育企业": ["国家级", "省级", "市级", "区县级"],
        "专精特新企业（培育）": ["国家级", "省级", "市级", "区县级"],
        "专精特新小巨人企业": ["国家级", "省级", "市级", "区县级"],
        "专业协会（IAOP）100强": ["国家级", "省级", "市级", "区县级"],
        "总部企业": ["国家级", "省级", "市级", "区县级"],
        "示范平台": ["国家级", "省级", "市级", "区县级"],
        "质量奖": ["国家级", "省级", "市级", "区县级"],
        "小巨人培育企业": ["国家级", "省级", "市级", "区县级"],
        "小巨人企业": ["国家级", "省级", "市级", "区县级"],
        "知识产权品牌服务机构": ["国家级", "省级", "市级", "区县级"],
        "资信等级": ["A", "AA", "AAA", "B", "C"],
        "增值业务电信许可证": [],
        "涉密信息系统集成资质": [],
        "信息系统工程监理单位": [],
        "国家信息安全测评和服务": [],
        "职业健康安全管理体系认证": [],
        "雏鹰（培育）企业": ["国家级", "省级", "市级", "区县级"]
    },
    "企业注册资金": "企业注册资金"
}
rc_data = ["魅力科技人物" , "魅力科技团队", "科学技术奖", "领军人才", "首席技师", "技能专家","“千人计划”人才",  "双创人才",  "科技企业家", "产业教授",
           "“百人计划”人才", "长江学者", "“金鸡湖人才计划”人才", "院士", "重点人才工程"]
area_regular_list = [
    "中国-江苏省-苏州市-虎丘区",
    "中国-江苏省-苏州市"
]
city_list = {
    "苏州": {
        "虎丘": {"leaf": 0},
        "高新": {"leaf": 0},
        "leaf": 1
    }
}

qyxz_list = ["科技型企业", "科技服务机构", "高校、科研院所", "社会团体", "金融机构"]

question_list = [
    "项目金额",
    "企业注册地区",
    "企业注册日期",
    "企业注册资金",
    "企业性质",
    "企业资质",
    "人才资质",
    "企业行业",
    "企业人数",
    "领军人数",
    "大专及以上人数",
    "本科及以上人数",
    "硕士及以上人数",
    "博士及以上人数",
    "总资产",
    "净利润",
    "营业收入",
    "税收",
    "知识产权总数",
    "专利总数",
    "注册商标总数",
    "软件著作权总数"
]

def search_area(input_area):
    cur = city_list
    while True:
        state = 0
        for k in cur:
            if k in input_area:
                cur = cur[k]
                state = 1
                break
        if state == 0:
            break
    return cur["leaf"]

# tika.initVM()
#
# parsed = parser.from_file('D:\data\\1611914203490.pdf')
# print(parsed["metadata"])
# print(parsed["content"])
data_content = """
    各设区市市场监管局、发展和改革委员会：

为加快推进质量诚信体系建设，推动质量信用分级分类监管，引导企业强化质量诚信意识，根据《省政府关于加快质量发展的意见》（苏政发〔2016〕88号）和《中共江苏省委、江苏省人民政府关于印发〈江苏省质量提升行动实施方案〉的通知》（苏发〔2018〕12号）精神，经研究，决定开展2021年度江苏省质量信用AA级及以上企业等级核定工作。现就有关事项通知如下：

一、申报范围

   凡注册地在江苏的企业（含港澳台地区），符合《工业企业质量信用评价》（DB32/T 1926-2011）的要求，经核定为质量信用A级等级满2年以上，可自愿申报质量信用AA级；经核定为质量信用AA级等级满1年以上，可自愿申报质量信用AAA级。

二、申报内容

申报企业填写《江苏省工业企业质量信用等级核定申请表》（登陆江苏省工业企业质量信用信息管理系统下载），并附相关证明材料。证明材料参考《江苏省工业企业质量信用现场核查报告》（见附件1）。

三、核定流程

1. 企业申报。符合条件的企业自7月7日至8月6日，登陆江苏省工业企业质量信用信息管理系统（http://222.190.96.186:8800/zlxy/login.jsp），提交网络申请（网络申请端口将于8月6日24：00关闭）；同时向所在地县（市、区）市场监管部门提交书面申请材料。

2. 组织上报。各县（市、区）市场监管局根据企业申请，指导企业进行网上申报，并对申报材料进行格式审查。

3. 资格初审。各设区市市场监管局负责对本区域内企业的申报材料进行初审，并于8月10日前将申报汇总表（见附件2）及书面申报材料一次性报省市场监管局质量发展处，邮箱1050753402@qq.com。

4. 信用审查。省发改委对申报企业的信用状况进行审查。

5. 材料审核。省市场监管局组织专家对申报材料进行审核。

6. 现场核查。省市场监管局将按照《江苏省工业企业质量信用现场核查报告》要求，组织对通过材料审核的企业进行现场核查。

7. 等级核定。现场核查符合AA级及以上条件的企业，省市场监管局将进行公示。经公示无异议的，由省市场监管局、省发展改革委联合发文核定为质量信用AAA、AA等级。

四、工作要求

1. 各地要高度重视，加大宣传发动力度，落实专门人员，加强对申报企业的指导，配合做好申报受理、现场核查等工作。

2. 各申报企业要按照规定填报申请材料，严禁弄虚作假。凡申报内容存在弄虚作假等问题的，将予以通报，三年内不得申报质量信用等级核定。企业质量信用等级核定申报承诺书（见附件3）请加盖公章与纸质材料一并提交给所在地市场监管部门。

3. 质量信用等级核定要坚持企业自愿原则，不收取企业任何费用。任何单位不得搭车收费，不得增加企业额外负担。
"""

key_scope = ["申报范围", "申报要求"]

class TwoHeadTree(object):

    def __init__(self, val=None):
        self.val = val
        self.level = -1
        self.next = dict()
        self.parent = None
        self.main_body = []
        self.is_direct_control = []

    def show(self):

        def dfs(tree):
            print("-"*tree.level+tree.val)
            for _, v in tree.next.items():
                dfs(v)
        dfs(self)

class Document(object):

    def __init__(self):
        self.content = []
        self.level = []
        self.title = [
            "^[一二三四五六七八九十]、.+",
            "^（[一二三四五六七八九十]）.+",
            "^[0-9]\..+",
            "^（[0-9]）.+",
            "^[①②③④⑤].+",
            "^[0-9]、.+",
            "^[0-9]{6} .+"
        ]
        # self.mrc_model = MRCMODULE()
        self.mrc_model = None

        self.root = TwoHeadTree("root")

    def title_ex(self, input_span):
        for i, title_p in enumerate(self.title):
            if re.match(title_p, input_span):
                return "title_{}".format(i)
        return None

    def parse_content(self, content):
        spans = content.split("\n")
        span_list = [{"title": "root", "content": [], "title_level": 0}]
        parent = dict()
        last_level = 0
        i = 0
        for span in spans:
            span = span.strip()
            if len(span) == 0:
                continue

            # print(span)

            title_ind = self.title_ex(span)
            if title_ind:

                if title_ind not in self.level:
                    self.level.append(title_ind)
                title_level = self.level.index(title_ind)
                # print(span_list[i])
                if last_level < title_level:
                    parent[i+1] = i
                elif last_level == title_level:
                    if last_level != 0:
                        parent_id = parent[i]
                        parent[i+1] = parent_id
                    else:
                        parent[i + 1] = 0
                else:
                    if title_level != 0:
                        # print(span_list[i])
                        cur_i = i
                        # print(cur_i, "-------")
                        while cur_i >=0 and span_list[cur_i]["title_level"] != title_level:
                            cur_i -= 1
                        # print(cur_i, "++++++", title_level)
                        if cur_i >= 0:
                            parent_id = parent[cur_i]
                            parent[i + 1] = parent_id
                last_level = title_level

                data = {
                        "title": span,
                        "title_level": title_level,
                        "content": []
                    }
                span_list.append(data)

                i += 1
            else:
                span_list[i]["content"].append(span)

        self.content = span_list
        self.parent = parent
        self.child = {}
        for c, p in self.parent.items():
            self.child.setdefault(p, [])
            self.child[p].append(c)

    # def get_title_num(self, ):


    def parse_content_v3(self, content):
        spans = content.split("\n")
        span_list = [{"title": "root", "content": [], "title_level": 0, "title_type": -1, "title_view": 0, "title_content": "root"}]
        parent = {}
        child = {0: []}
        i = 0
        type_title_6_ind = 1
        for span in spans:
            span = span.strip()
            if len(span) == 0:
                continue
            if span[:3] == "附件：" or span[:3] == "附件:":
                span_list[i]["content"].append(span[:3])
                span = span[3:].strip()

            span = span.replace("(", "（").replace(")", "）")
            span = span.replace("．", ".")
            title_ind = self.title_ex(span)
            if title_ind == "title_5":
                title_ind = "title_2"
                span = span.replace("、", ".")

            if title_ind is None:
                span_list[i]["content"].append(span)
            else:

                child.setdefault(i + 1, [])
                title_ind = int(title_ind.split("_")[1])
                span_list.append(
                    {"title": span, "content": [], "title_level": -1, "title_type": title_ind, "title_view": 0, "title_content": ""})
                if title_ind == 0:
                    ind = span.split("、")[0]
                    span_list[i+1]["title_content"] = span.split("、")[1]
                elif title_ind == 1:
                    dt = re.search("（(.+?)）", span)
                    if dt:
                        ind = dt.group(1)
                    span_list[i + 1]["title_content"] = "）".join(span.split("）")[1:])
                elif title_ind == 2:
                    ind = span.split(".")[0]
                    span_list[i + 1]["title_content"] = span.split(".")[1]
                elif title_ind == 3:
                    dt = re.search("（([1-9]+)?）", span)
                    if dt:
                        ind = dt.group(1)
                    span_list[i + 1]["title_content"] = "）".join(span.split("）")[1:])
                elif title_ind == 4:
                    ind = span[0]
                    span_list[i + 1]["title_content"] = span[1:]
                elif title_ind == 5:
                    ind = span.split("、")[0]
                    span_list[i + 1]["title_content"] = span.split("、")[1]
                elif title_ind == 6:
                    ind = str(type_title_6_ind)
                    type_title_6_ind += 1
                    span_split = [x for x in span.split(" ") if x]
                    span_list[i + 1]["title_content"] = span_split[1]
                else:
                    print(span, title_ind)
                    raise Exception()
                # print("st ", ind)
                if ind in ["1", "一", "①"]:
                    if span_list[i]["title_type"] == span_list[i+1]["title_type"]:
                        parent[i + 1] = parent[i]
                        child[parent[i]].append(i + 1)
                        span_list[i + 1]["title_level"] = span_list[i]["title_level"]
                    else:
                        parent[i+1] = i
                        child[i].append(i+1)
                        span_list[i+1]["title_level"] = span_list[i]["title_level"]+1
                else:
                    j = i
                    while j >= 0:
                        if span_list[j]["title_type"] == title_ind:
                            break
                        j -= 1
                    if j == -1:
                        print(span)
                        print(span_list)
                        raise Exception
                        # parent[i + 1] = i
                        # child[i].append(i + 1)
                        # span_list[i + 1]["title_level"] = span_list[i]["title_level"] + 1
                    else:
                        parent[i + 1] = parent[j]
                        child[parent[j]].append(i+1)
                        span_list[i + 1]["title_level"] = span_list[j]["title_level"]
                i += 1
        self.content = span_list

        seed = [(0, self.root)]
        while seed:
            pi, cur_tree = seed.pop(0)
            for c in child.get(pi, []):
                next_tree = TwoHeadTree(c)
                cur_tree.next[c] = next_tree
                next_tree.parent = pi
                seed.append((c, next_tree))

    def parse_half_struction(self):
        conditions = []
        project_names = []
        text_materials = []
        report_processes = []
        project_infos = []
        contact_infos = []
        benefit_infos = []
        short_name = []
        apply_dates = []
        def check_score(title_list):
            if len(title_list) ==0:
                return 0.0
            key_title_list = ["申报时间", "申报类型", "申报要求", "申报材料", "申报流程", "入库企业范围", "企业入库流程",
                              "申报对象条件", "申报材料", "资助方式及标准", "申请类别", "重点产品领域"]
            score = 0.0
            for title in title_list:
                if title in key_title_list:
                    score += 1
            return score/len(title_list)

        for pre_content in self.content[0]["content"]:
            rs = re.search("、([\u4e00-\u9fa50-9]+?)（以下简称(.+?)）", pre_content)
            if rs:
                short_name.append((rs.group(1), rs.group(2)))
        for c, child in self.root.next.items():
            span = self.content[c]
            if span["title_content"] == "申报要求":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
                elif if_next_node == len(child.next):
                    state = 0
                    for ci, childi in child.next.items():
                        if self.content[ci]["title_content"] in ["申报条件", "申报对象"]:
                            state += 1
                            for cii, childii in childi.next.items():
                                conditions.append(self.content[cii]["title"])
                        elif self.content[ci]["title_content"] == "申报类别":
                            for cii, childii in childi.next.items():
                                project_names.append(self.content[cii]["title_content"])
            elif span["title_content"] == "申报条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == len(child.next):
                    for ci, childi in child.next.items():
                        project_names.append(self.content[ci]["title_content"])
                        project_info = {
                            "project_name": self.content[ci]["title_content"],
                            "conditions": []
                        }
                        for cii, childii in childi.next.items():
                            project_info["conditions"].append(self.content[cii]["title"])
                        project_infos.append(project_info)
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "申报材料":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    text_materials = self.content[c]["content"]
                elif if_next_node == len(child.next):
                    for ci, childi in child.next.items():
                        project_names.append(self.content[ci]["title_content"])
                        project_info = {
                            "project_name": self.content[ci]["title_content"],
                            "text_material": []
                        }
                        for cii, childii in childi.next.items():
                            project_info["text_material"].append(self.content[cii]["title"])
                        text_materials.append(project_info)
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        text_materials.append(self.content[ci]["title"])

            elif span["title_content"] == "申报程序":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == 0:
                    for ci, childi in child.next.items():
                        report_processes.append(self.content[ci]["title"])
            elif span["title_content"] == "申报流程":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    report_processes = self.content[c]["content"]
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        report_processes.append(self.content[ci]["title"])
            elif span["title_content"] == "申报方式":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == len(child.next):
                    for ci, childi in child.next.items():
                        project_names.append(self.content[ci]["title_content"])
                        project_info = {
                            "project_name": self.content[ci]["title_content"],
                            "report_process": []
                        }
                        for cii, childii in childi.next.items():
                            project_info["report_process"].append(self.content[cii]["title"])
                        report_processes.append(project_info)
            elif span["title_content"] == "可申报项目":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == 0:
                    for ci, childi in child.next.items():
                        project_names.append(self.content[ci]["title_content"])
            elif span["title_content"] == "推荐要求":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "工作要求":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                for sub_content in self.content[c]["content"]:
                    conditions.append(sub_content)
                if if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "申报内容":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == 0:
                    for ci, childi in child.next.items():
                        project_names.append(self.content[ci]["title_content"])
            elif span["title_content"] == "申报项目类别":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        project_names.append(self.content[ci]["title_content"])
            elif span["title_content"] == "申报事项":
                if_next_node = 0
                if_next_value = []
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if self.content[ci]["title_content"][-7:] == "符合下列条件：":
                            if_next_value.append(ci)
                        if_next_node += 1
                if if_next_node and len(if_next_value):
                    for ci, childi in child.next.items():
                        if len(childi.next):
                            if self.content[ci]["title_content"][-7:] == "符合下列条件：":
                                for cii, childii in childi.next.items():
                                    conditions.append(self.content[cii]["title"])
                        else:
                            conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "申报主体":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    for sub_content in self.content[c]["content"]:
                        conditions.append(sub_content)
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "支持类别":
                for ci, childi in child.next.items():
                    project_names.append(self.content[ci]["title_content"])
            elif span["title_content"] == "联系电话":
                pass
            elif span["title_content"] == "申报范围":
                for child_content in span["content"]:
                    conditions.append(child_content)
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "基本条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "申报方式":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        report_processes.append(self.content[ci]["title"])
            elif span["title_content"] == "项目申报企业（单位）必须符合下列条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "认定条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "申报内容及申报材料":
                for ci, childi in child.next.items():
                    sub_title = self.content[ci]["title_content"]
                    if check_is_project_title(sub_title):
                        project_info = {
                            "project_name": sub_title
                        }
                        for cii, childii in childi.next.items():
                            if self.content[cii]["title_content"] == "此次申报申报企业须符合以下条件:":
                                project_info["conditions"] = self.content[cii]["content"]
                                for ciii, childiii in childii.next.items():
                                    project_info["conditions"].append(self.content[ciii]["title"])
                            elif self.content[cii]["title_content"] == "申报材料":
                                project_info["text_materials"] = []
                                for ciii, childiii in childii.next.items():
                                    project_info["text_materials"].append(self.content[ciii]["title"])
                        project_infos.append(project_info)
            elif span["title_content"] == "补助标准":
                for ci, childi in child.next.items():
                    sub_title = self.content[ci]["title_content"]
                    benefit_infos.append(sub_title)
            elif span["title_content"] == "补贴内容":
                for ci, childi in child.next.items():
                    sub_title = self.content[ci]["title_content"].strip()
                    benefit_infos.append(sub_title)
            elif span["title_content"] == "申报资格条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
            elif span["title_content"] == "评选要求":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == len(child.next):
                    for ci, childi in child.next.items():
                        project_info = {
                            "project_name": self.content[ci]["title_content"],
                            "conditions": []
                        }
                        for cii, childii in childi.next.items():
                            sub_titile_content = self.content[cii]["title_content"]
                            if sub_titile_content in ["申报主体", "申报作品要求", "评选条件"]:
                                for sub_content in self.content[cii]["content"]:
                                    project_info["conditions"].append(sub_content)
                                for ciii, childiii in childii.next.items():
                                    project_info["conditions"].append(self.content[ciii]["title"])
                        project_infos.append(project_info)

            elif span["title_content"] == "申报时间":
                for sp in self.content[c]["content"]:
                    apply_dates.append(sp)
            elif span["title_content"] == "工作程序":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if if_next_node == 0:
                    for ci, childi in child.next.items():
                        report_processes.append(self.content[ci]["title"])
            elif span["title_content"] == "申报对象":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next)==0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        if "：" in  sub_titile_content and sub_titile_content.index("：")<20:
                            project_info = {
                                "project_name": sub_titile_content.split("：")[0],
                                "conditions": [sub_titile_content]
                            }
                            project_infos.append(project_info)
                        else:
                            conditions.append(sub_titile_content)
            elif span["title_content"] == "支持重点和申报条件":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_type"] == 6:
                        project_names.append(self.content[ci]["title_content"])
            elif span["title_content"] == "申报基本条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        conditions.append(sub_titile_content)
            elif span["title_content"] == "申报机构范围":
                if len(child.next) == 0:
                    for sub_content in self.content[c]["content"]:
                        conditions.append(sub_content)
            elif span["title_content"] == "遴选要求":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        conditions.append(sub_titile_content)
            elif span["title_content"] == "资助对象":
                if len(child.next) == 0:
                    for sub_content in self.content[c]["content"]:
                        conditions.append(sub_content)
            elif span["title_content"] == "支持类型和条件":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_type"] == 6:
                        project_info = {
                            "project_name": self.content[ci]["title_content"],
                            "conditions": []
                        }
                        for sub_content in self.content[ci]["content"]:
                            project_info["conditions"].append(sub_content)
                        for cii, childii in childi.next.items():
                            project_info["conditions"].append(self.content[cii]["title_content"])
                        project_infos.append(project_info)
            elif span["title_content"] == "申报对象和条件":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        conditions.append(sub_titile_content)
            elif span["title_content"] == "资助标准":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        benefit_infos.append(sub_titile_content)
            elif span["title_content"] == "联系方式":
                for sub_content in span["content"]:
                    contact_infos.append(sub_content)
            elif span["title_content"] == "评定对象和条件":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"] == "评定条件":
                        for cii, childii in childi.next.items():
                            conditions.append(self.content[cii]["title"])

        # 标题子目录问题
        if len(conditions) == 0 and len(project_names) == 0 and len(text_materials) == 0 and len(report_processes) == 0:
            state = 0
            for c, child in self.root.next.items():
                span = self.content[c]
                title_content= span["title_content"]
                if len(title_content) > 30:
                    if title_content[:4] == "申报要求":
                        conditions.append(title_content)
                        state += 1
                    elif title_content[:4] == "申报主体":
                        conditions.append(title_content)
                        state += 1
                    elif title_content[:4] == "申报限制":
                        conditions.append(title_content)
                        state += 1
                    elif title_content[:4] == "申报程序":
                        report_processes.append(title_content)
                        state += 1
            if state == 0:
                for c, child in self.root.next.items():

                    span = self.content[c]
                    title_list = []
                    for k, kchild in child.next.items():
                        sub_span = self.content[k]["title_content"]
                        title_list.append(sub_span)
                    score = check_score(title_list)
                    if score > 0.6:
                        project_names.append(span["title_content"])
                        project_info = {
                            "project_name": span["title_content"],
                            "text_materials": [],
                            "conditions": [],
                            "apply_dates": []
                        }
                        for k, kchild in child.next.items():
                            sub_span = self.content[k]["title_content"]
                            if sub_span == "申报材料":
                                if_next_node = 0
                                for ci, childi in kchild.next.items():
                                    if len(childi.next):
                                        if_next_node += 1
                                if if_next_node == 0:
                                    project_info["text_materials"] = self.content[k]["content"]
                            elif sub_span == "申报要求":
                                if_next_node = 0
                                for ci, childi in kchild.next.items():
                                    if len(childi.next):
                                        if_next_node += 1

                                if len(kchild.next) == 0:
                                    project_info["conditions"] = self.content[k]["content"]
                                elif if_next_node == 0:
                                    for ci, childi in kchild.next.items():
                                        project_info["conditions"].append(self.content[ci]["title"])


                            elif sub_span == "入库企业范围":
                                if_next_node = 0
                                for ci, childi in kchild.next.items():
                                    if len(childi.next):
                                        if_next_node += 1
                                if if_next_node == 0:
                                    for ci, childi in kchild.next.items():
                                        project_info["conditions"].append(self.content[ci]["title"])
                            elif sub_span == "申报时间":
                                for sp in self.content[k]["content"]:
                                    project_info["apply_dates"].append(sp)
                        project_infos.append(project_info)

        res = {
            "conditions": conditions,
            "project_names": project_names,
            "text_materials": text_materials,
            "report_processes": report_processes,
            "project_infos": project_infos,
            "short_name": short_name,
            "benefit_infos": benefit_infos,
            "apply_dates": apply_dates,
            "contact_infos": contact_infos
        }
        self.half_struction = res

        return res

    def parse_precision(self, conditions):
        res = {
            "zhibiao": [],
            "zhibiao_person": []
        }
        for condition in conditions:
            if "注册" in condition:
                rs = re.search("在(.{1,10}?)注册", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("注册地在(.+?)范围内", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("申报单位应在(.+?)设立", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("登记注册在(.+?);", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("注册在(.+?)内", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("注册地在(.+?)的企业（(.+?)）", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                    res["zhibiao"].append(("企业注册地址", rs.group(2)))
                rs = re.search("注册([0-9]+?年以上)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册日期", rs.group(1)))
                rs = re.search("注册登记、税务登记和主要工作场所均在([\u4e00-\u9fa5]+)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("为([\u4e00-\u9fa5]+)登记注册", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("注册地在([\u4e00-\u9fa5]+)的企业", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("注册登记.税务登记和主要工作场所均在([\u4e00-\u9fa5]+)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("正常经营([0-9]+年以上)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册日期", rs.group(1)))
                rs = re.search("工商注册地、税务征管关系及统计关系在([\u4e00-\u9fa5]+)范围内", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                rs = re.search("注册成立([一二三四五六七八九十]年（365个日历天数）以上)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册日期", rs.group(1)))
            if "成立" in condition:
                rs = re.search("在(.+?)成立([一二三四五六]年以上)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))
                    res["zhibiao"].append(("企业注册日期", rs.group(2)))
                rs = re.search("企业成立([一二三四五六七八九十]年以内)", condition)
                if rs:
                    res["zhibiao"].append(("企业注册日期", rs.group(1)))
            if "设立" in condition:
                rs = re.search("在([\u4e00-\u9fa5]+)依法设立", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地址", rs.group(1)))

            if "营业收入" in condition:
                rs = re.search("营业收入([0-9]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("营业收入", rs.group(1)))
                else:
                    rs = re.search("营业收入总额在([0-9]+?万元)以上", condition)
                    if rs:
                        res["zhibiao"].append(("营业收入", rs.group(1)))
            if "利税" in condition:
                rs = re.search("利税总额([0-9]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("税收", rs.group(1)))
            if "三年营业收入平均增长率" in condition:
                rs = re.search("三年营业收入平均增长率([0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("三年营业收入平均增长率", rs.group(1)))
            if "平均营业利润率" in condition:
                rs = re.search("平均营业利润率([0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("平均营业利润率", rs.group(1)))
            if "发明专利" in condition:
                rs = re.search("拥有([0-9]+?项及以上)发明专利", condition)
                if rs:
                    res["zhibiao"].append(("发明专利", rs.group(1)))
            if "专利总数" in condition:
                rs = re.search("专利总数(不少于[0-9]+?项)", condition)
                if rs:
                    res["zhibiao"].append(("发明专利", rs.group(1)))
            if "软件著作权" in condition:
                rs = re.search("软件著作权[\u4e00-\u9fa5、]+(不少于[0-9]+?项)", condition)
                if rs:
                    res["zhibiao"].append(("软件著作权", rs.group(1)))
            if "软件产品评估" in condition:
                rs = re.search("软件产品评估[\u4e00-\u9fa5、]+(不少于[0-9]+?项)", condition)
                if rs:
                    res["zhibiao"].append(("软件产品评估", rs.group(1)))
            if "技术中心工作人员" in condition:
                rs = re.search("技术中心工作人员数(不低于[0-9]+?人)", condition)
                if rs:
                    res["zhibiao"].append(("技术中心工作人员", rs.group(1)))
            if "年主营业务收入" in condition:
                rs = re.search("年主营业务收入(不低于[0-9\s]+?亿元)", condition)
                if rs:
                    res["zhibiao"].append(("营业收入", rs.group(1)))
            if "研究开发投入" in condition:
                rs = re.search("支出额(不低于[0-9\s]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("研究开发投入", rs.group(1)))
            if "技术带头人" in condition:
                rs = re.search("人员数不少于([0-9]+?人)", condition)
                if rs:
                    res["zhibiao"].append(("技术中心工作人员", rs.group(1)))

            if "企业实收资本" in condition:
                rs = re.search("企业实收资本(不低于[0-9]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("总资产", rs.group(1)))
            if "研发费用总额占同期销售收入总额的比例" in condition:
                if "比例符合以下标准：" in condition:
                    zb = []
                    rs = re.findall("销售收入为([0-9亿万—以下上元]+)的?企业，比例(不低于[0-9]+?％)", condition)
                    for r in rs:
                        zb.append({"销售收入": r[0], "比例": r[1]})
                    if zb:
                        res["zhibiao"].append(("研发费用总额占同期销售收入总额的比例", zb))
            if "研发投入占营业收入比例" in condition:
                rs = re.search("研发投入占营业收入比例(不低于[0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("研发投入占营业收入比例", rs.group(1)))
            if "高新技术企业" in condition:
                res["zhibiao"].append(("企业资质", "高新技术企业"))
            if "尚未在主板.创业板.科创板上市或在“新三板”挂牌" in condition:
                res["zhibiao"].append(("企业资质", "未上市"))
            if "失信" in condition:
                rs = re.search("(近[0-9]+?年)无严重失信行为", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", rs.group(1)))
                rs = re.search("近([一二两三四五六七八九]年内?)无严重失信行为", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", rs.group(1)))
                rs = re.search("近([0-9]+?年)信用状况良好，无严重失信行为", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", rs.group(1)))
                rs = re.search("近([一二两三四五六七八九]年内?)信用状况良好，无重大失信记录", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", rs.group(1)))
            if "年度研究与试验发展经费" in condition:
                rs = re.search("年度研究与试验发展经费支出额(不低于[0-9]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("研究开发投入", rs.group(1)))

            if "研究与试验发展人员" in condition:
                rs = re.search("研究与试验发展人员数不低于([0-9\s]+?人)", condition)
                if rs:
                    res["zhibiao"].append(("研发人数", rs.group(1)))
            if "上市公司" in condition:
                res["zhibiao"].append(("企业资质", "上市公司"))
            if "农机（农业）企业" in condition:
                res["zhibiao"].append(("企业性质", "农业"))
            if "属于科技型企业" in condition:
                res["zhibiao"].append(("企业性质", "科技"))
            if "净资产总额" in condition:
                rs = re.search("净资产总额(不超过[0-9]+?[万亿]元)", condition)
                if rs:
                    res["zhibiao"].append(("总资产", rs.group(1)))
            if "销售总额" in condition:
                rs = re.search("销售总额(不超过[0-9]+?[万亿]元)", condition)
                if rs:
                    res["zhibiao"].append(("营业收入", rs.group(1)))
            if "拥有对外贸易经营" in condition:
                res["zhibiao"].append(("企业性质", "对外贸易"))
            if "从事文化和旅游装备技术研发" in condition:
                res["zhibiao"].append(("企业性质", "文化和旅游"))
            if "核定为质量信用" in condition:
                rs = re.findall("核定为质量信用([ABC]+?级等级)满([0-9]?年以上)", condition)
                for sub in rs:
                    res["zhibiao"].append(("企业资质", sub))
            if "工业企业质量信用评价" in condition:
                res["zhibiao"].append(("企业性质", "工业、制造业"))
            if "农业" in condition:
                res["zhibiao"].append(("企业性质", "农业"))
            if "具有知识产权" in condition:
                res["zhibiao"].append(("知识产权总数", "大于等于1"))
            if "知识产权运营服务机构持有可运营专利" in condition:
                res["zhibiao"].append(("知识产权总数", "大于等于1"))
                res["zhibiao"].append(("专利总数", "大于等于1"))
            if "金融机构" in condition:
                res["zhibiao"].append(("企业性质", "金融"))
            if "拥有自主有效的国内注册商标" in condition:
                res["zhibiao"].append(("注册商标总数", "大于等于1"))
            if "外资总部型企业" in condition:
                res["zhibiao"].append(("企业性质", "外资总部型"))
            if "省级跨国公司地区总部" in condition:
                res["zhibiao"].append(("企业性质", "跨国公司地区总部"))
            tp = re.search("详见(《[\S]+?》)", condition)
            if tp:
                res["zhibiao"].append(("链接资料", tp.group(1)))
            if "“独角兽”培育企业库" in condition:
                res["zhibiao"].append(("企业资质", "独角兽"))
            if "人工智能场景应用重点领域" in condition:
                res["zhibiao"].append(("企业性质", "人工智能"))
            tp = re.search("申报项目必须符合(.+?)（", condition)
            if tp:
                res["zhibiao"].append(("链接资料", tp.group(1)))
            tp = re.findall("满足(《[\u4e00-\u9fa5]+?》)和(《[\u4e00-\u9fa5]+?》)中明确的具体条件", condition)
            for sut in tp:
                res["zhibiao"].append(("链接资料", sut))
            tp = re.findall("满足(《[\u4e00-\u9fa5]+?》)中明确的具体条件", condition)
            for sut in tp:
                res["zhibiao"].append(("链接资料", sut))
            if "学位" in condition:
                rs = re.search("具有(.+?)学位", condition)
                if rs:
                    res["zhibiao_person"].append(("学位", rs.group(1)))
                rs = re.search("拥有(.+?)学位", condition)
                if rs:
                    res["zhibiao_person"].append(("学位", rs.group(1)))
            if "股权" in condition:
                rs = re.search("股权一般(不低于[0-9]+%)", condition)
                if rs:
                    res["zhibiao_person"].append(("股权", rs.group(1)))
            if "现金出资" in condition:
                rs = re.search("现金出资（实收资本，不含技术入股）(不少于[0-9]+万元)", condition)
                if rs:
                    res["zhibiao_person"].append(("现金出资", rs.group(1)))
            if "年龄" in condition:
                rs = re.search("年龄一般(不超过[0-9]+?周岁)", condition)
                if rs:
                    res["zhibiao_person"].append(("年龄", rs.group(1)))
                rs = re.search("年龄(不超过[0-9]+?周岁)", condition)
                if rs:
                    res["zhibiao_person"].append(("年龄", rs.group(1)))
            if "引才企业应具备以下条件" in condition:
                if "国家高新技术企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "国家高新技术企业"))
                if "市级高成长性创新型企业培育计划" in condition:
                    res["zhibiao_person"].append(("企业资质", "市级高成长性创新型企业培育计划"))
                if "瞪羚企业培育计划入选企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "瞪羚企业培育计划入选企业"))
                if "市级以上领军人才创办的企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "市级以上领军人才创办的企业"))
                if "市级及以上研发机构的企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "市级及以上研发机构的企业"))
            tp = re.search("(《.+?》)明确的具体申报条件", condition)
            if tp:
                res["zhibiao"].append(("链接资料", tp.group(1)))
            tp = re.search("(《.+?》)规定的具体条件", condition)
            if tp:
                res["zhibiao"].append(("链接资料", tp.group(1)))
            tp = re.search("(《.+?》)明确的具体条件", condition)
            if tp:
                res["zhibiao"].append(("链接资料", tp.group(1)))
            if "研究开发费用占销售收入的比重" in condition:
                rs = re.search("研究开发费用占销售收入的比重(不低于[0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("研究开发费用占销售收入的比重", rs.group(1)))
            if "中小微企业" in condition:
                res["zhibiao"].append(("企业资质", "中小微企业"))
            if "研发经费支出占主营业务收入" in condition:
                rs = re.search("研发经费支出占主营业务收入的比重应(高于[0-9]+%)", condition)
                if rs:
                    res["zhibiao"].append(("研发经费支出占主营业务收入", rs.group(1)))
            if "企业未上市" in condition:
                res["zhibiao"].append(("企业资质", "未上市"))
            if "入选苏州市瞪羚计划企业" in condition:
                res["zhibiao"].append(("企业资质", "瞪羚计划企业"))
            if "电子商务年交易额" in condition:
                rs = re.search("电子商务年交易额达到([0-9]+?亿元以上)", condition)
                if rs:
                    res["zhibiao"].append(("电子商务年交易额", rs.group(1)))
            if "电子商务服务业年营业收入" in condition:
                rs = re.search("电子商务服务业年营业收入达到([0-9]+?万元以上)", condition)
                if rs:
                    res["zhibiao"].append(("电子商务服务业年营业收入", rs.group(1)))
            if "投入运营" in condition:
                rs = re.search("正式投入运营([0-9]+?年以上)", condition)
                if rs:
                    res["zhibiao"].append(("经营时间", rs.group(1)))
            if "国内外知名企业、高校、科研单位从事研发及管理经历" in condition:
                rs = re.search("有(.+?)国内外知名企业、高校、科研单位从事研发及管理经历", condition)
                if rs:
                    res["zhibiao_person"].append(("国内外知名企业、高校、科研单位从事研发及管理经历", rs.group(1)))
            if "昆山市级及以上人才计划认定" in condition:
                res["zhibiao_person"].append(("昆山市级及以上人才计划"))
            if "项目总投入" in condition:
                rs = re.search("项目总投入(不少于[.+?]万元)", condition)
                if rs:
                    res["zhibiao_person"].append(("项目总投入", rs.group(1)))
            if "研发、管理工作经历或自主创业经历" in condition:
                rs = re.search("具有([0-9]+?年以上)相关研发、管理工作经历或自主创业经历", condition)
                if rs:
                    res["zhibiao_person"].append(("相关研发、管理工作经历或自主创业经历", rs.group(1)))
            if "具有自主知识产权" in condition:
                res["zhibiao_person"].append(("知识产权总数", "大于等于1"))
            if "税前年薪" in condition:
                rs = re.search("税前年薪(不低于([0-9]+?)万元)", condition)
                if rs:
                    res["zhibiao_person"].append(("税前年薪", rs.group(1)))
            if "苏州市瞪羚计划企业" in condition:
                res["zhibiao_person"].append(("企业资质", "苏州市瞪羚计划企业"))
            if "苏州市“独角兽”培育企业" in condition:
                res["zhibiao_person"].append(("企业资质", "苏州市“独角兽”培育企业"))
            if "注册资本" in condition:
                rss = re.findall("项目注册资本(不低于[0-9]+?万元)", condition)
                for rs in rss:
                    res["zhibiao"].append(("注册资本", rs))
            if "实缴出资额" in condition:
                rs = re.search("实缴出资额(不少于[0-9]+?万元)", condition)
                if rs:
                    res["zhibiao_person"].append(("实缴出资额", rs.group(1)))
            if "内容详见" in condition:
                rs = re.search("内容详见(.+?)。", condition)
                if rs:
                    res["zhibiao_person"].append(("链接文件", rs.group(1)))
            if "注册资金" in condition:
                rs = re.search("注册资金在([0-9]+?万元人民币以上)", condition)
                if rs:
                    res["zhibiao"].append(("注册资本", rs.group(1)))
            if "职工人数" in condition:
                rs = re.search("职工人数一般在([0-9]+?人以上)", condition)
                if rs:
                    res["zhibiao"].append(("企业人数", rs.group(1)))
            if "生产经营" in condition:
                rs = re.search("生产经营([0-9]+?年以上)", condition)
                if rs:
                    res["zhibiao"].append(("经营时间", rs.group(1)))
            if "企业成立日期" in condition:
                rs = re.search("企业成立日期在([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日（含）以后)", condition)
                if rs:
                    res["zhibiao"].append(("企业成立日期", rs.group(1)))
                rs = re.search("企业成立日期在([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日（含）-[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日（含）之间)", condition)
                if rs:
                    res["zhibiao"].append(("企业成立日期", rs.group(1)))
            if "最后一轮投后估值" in condition:
                rs = re.search("最后一轮投后估值(.+?)，", condition)
                if rs:
                    res["zhibiao"].append(("最后一轮投后估值", rs.group(1)))
            if "最新一轮投后估值" in condition:
                rs = re.search("最新一轮投后估值([0-9\-]+亿美元或[0-9\-]+亿元人民币)", condition)
                if rs:
                    res["zhibiao"].append(("最新一轮投后估值", rs.group(1)))
            if "未在主板或创业板上市" in condition:
                res["zhibiao"].append(("企业资质", "未上市"))
            if "上年度销售收入" in condition:
                rs = re.search("企业上年度销售收入（不含流水.关联交易）(不低于[0-9]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("上年度销售收入", rs.group(1)))
            if "三年销售收入" in condition and "净利润平均增长率" in condition:
                rs = re.search("三年销售收入（不含流水.关联交易）或净利润平均增长率(不低于[0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("三年销售收入增长率", rs.group(1)))
                    res["zhibiao"].append(("净利润平均增长率", rs.group(1)))
            if "业务收入" in condition:
                rs = re.search("([0-9]{4}年度)相关业务收入应(不低于[0-9]+?万元)", condition)
                if rs:
                    res["zhibiao"].append(("业务收入", (rs.group(1), rs.group(2))))
            # if "年龄" in condition:
            #     rs = re.search("税前年薪(不低于([0-9]+?)万元)", condition)
            #     if rs:
            #         res["zhibiao_person"].append(("年龄", rs.group(1)))



        print(res)

    def parse_contact_info(self):
        contact_infos = []
        for sub_sentence in self.half_struction["contact_infos"]:
            print([sub_sentence])
            rs = re.search("([\u4e00-\u9fa5]\u2003[\u4e00-\u9fa5])+\s([0-9]{8})", sub_sentence)
            if rs:
                contact_infos.append((rs.group(1), rs.group(2)))
        print(contact_infos)
        return contact_infos

    # 如果数据就是一坨
    def parse_root(self):
        for span in self.content:
            for span_content in span["content"]:
                span_content_list = span_content.split("。")
                for sentence in span_content_list:
                    if "注册" in sentence:
                        res = self.mrc_model.mrc("企业注册地区", sentence)
                        if res:
                            print(res, sentence)

    def display_document(self):
        for span in self.content:
            print("\t"*span["title_level"] + span["title"])

    # 粗分类
    def parse_content_v2(self):
        condition_sentence = []
        cailiao_sentence = []
        jiangli_sentence = []

        for i, span in enumerate(self.content):
            title = span["title"]
            if "申报条件" in title or "符合下列条件" in title or "申报要求" in title or "申报主体" in title or "申报限制" in title or "评价对象" in title or \
                    "申报资格条件" in title or "申报范围" in title or "申报基本标准" in title or "申报作品要求" in title or "评选条件" in title:
                for sentence in span["content"]:
                    condition_sentence.append(sentence)

                child_info = self.child.get(i, [])
                for ci in child_info:
                    title = self.content[ci]["title"]
                    condition_sentence.append(title)

            title = span["title"]
            if "申报材料" in title:
                for sentence in span["content"]:
                    cailiao_sentence.append(sentence)

                child_info = self.child.get(i, [])
                for ci in child_info:
                    title = self.content[ci]["title"]
                    cailiao_sentence.append(title)
            title = span["title"]
            if "补贴内容" in title:
                for sentence in span["content"]:
                    jiangli_sentence.append(sentence)

                child_info = self.child.get(i, [])
                for ci in child_info:
                    title = self.content[ci]["title"]
                    jiangli_sentence.append(title)

        # 筛选流程
        def f(input_str):
            if len(input_str) < 5:
                return False
            if "申报材料" in input_str or "申报时间" in input_str:
                if len(input_str) < 10:
                    return False
            return True

        condition_sentence = [st for st in condition_sentence if f(st)]
        print("条件")
        print(condition_sentence)
        print("材料")
        print(cailiao_sentence)
        print("奖励")
        print(jiangli_sentence)

    def parse_content_v1(self):
        zhuce_list = []
        zc_list = set()
        shenbaocailiao_list = []
        sbcl_list = set()
        for i, span in enumerate(self.content):
            title = span["title"]

            if "申报条件" in title or "符合下列条件" in title or "申报要求" in title or "申报主体" in title:
                if "注册" in title:
                    zhuce_list.append(title)
                for sentence in span["content"]:
                    if "注册" in sentence:
                        zhuce_list.append(sentence)
                child_info = self.child.get(i, [])
                for ci in child_info:
                    title = self.content[ci]["title"]
                    content = self.content[ci]["content"]
                    if "注册" in title:
                        zhuce_list.append(title)
                    for sentence in content:
                        if "注册" in sentence:
                            zhuce_list.append(sentence)
            if "申报方式" in title:
                for sentence in span["content"]:
                    sentence_list = sentence.split("。")
                    for sent in sentence_list:
                        if "材料" in sent:
                            shenbaocailiao_list.append(sent)
                child_info = self.child.get(i, [])
                for ci in child_info:
                    title = self.content[ci]["title"]
                    content = self.content[ci]["content"]
                    if "材料" in title:
                        shenbaocailiao_list.append(title)
                    for sentence in content:
                        if "材料" in sentence:
                            shenbaocailiao_list.append(sentence)
        for sentence in zhuce_list:
            # print(sentence)
            ans = self.mrc_model.mrc("企业注册地区、地址", sentence)
            if ans:
                zc_list.add(ans)
        for sentence in shenbaocailiao_list:
            ans = self.mrc_model.mrc("申报 材料", sentence)
            if ans:
                sbcl_list.add(ans)
        print(zc_list)
        print(sbcl_list)


    def extract_zb(self):
        condition_title_level = -1
        target_list = dict()
        for i, span in enumerate(self.content):
            title = span["title"]

            if "申报范围" in title or "申报条件" in title or "申报要求" in title or "申报对象" in title or "申报内容" in title:
                # parent = self.content[]["title"]
                parent_id = self.parent[i]
                target_list.setdefault(parent_id, [])
                target_list[parent_id].append((i, title))
                # for span_text in span["content"]:
                #     print("context: ", span_text)
                #     for qus in query_items:
                #         ans = mrc(qus, span_text)
                #         if len(ans) > 10:
                #             continue
                #         print(qus, ans)

        # print(target_list)
        # def clear_title(input_str, ind):
        #     if self.content[ind]["title_level"] == 0:
        #         return input_str.split("")
        # print(self.parent)
        res = []
        for target, condition_list in target_list.items():
            target_info = {
                "target_name": self.content[target]["title"],
                # "企业注册地区": "",
                "成立时间": "",
                "企业资质": "",
                "人才资质": "",
                "企业注册资金": "",
                "企业性质": "",
                "营业收入": ""
            }
            zz_list = []
            zc_list = set()
            rc_list = []
            clsj_list = []
            qyxz_list = []
            yysr_list = []
            for content in self.content[target]["content"]:
                for dt in rc_data:
                    if dt in content:
                        rc_list.append(dt)

            for i, x in condition_list:
                child_list = []
                if i in self.child:
                    child_list = self.child[i]

                for content in self.content[i]["content"]:
                    for z, _ in query_item_list["企业资质"].items():
                        if z in content:
                            zz_list.append(z)

                        for c in child_list:
                            if z in self.content[c]["title"]:
                                zz_list.append(z)
                            if "注册" in self.content[c]["title"]:
                                ans = self.mrc_model.mrc("企业注册地区、地址", self.content[c]["title"])
                                if ans:
                                    zc_list.add(ans)

                            for content in self.content[c]["content"]:
                                if z in content:
                                    zz_list.append(z)
                                if "注册" in content:
                                    ans = self.mrc_model.mrc("企业注册地区、地址", content)
                                    if ans:
                                        zc_list.add(ans)
                                for dt in rc_data:
                                    if dt in content:
                                        rc_list.append(dt)
                    if "注册" in content:
                        ans = self.mrc_model.mrc("企业注册地区、地址", content)
                        if ans:
                            zc_list.add(ans)
                    for dt in rc_data:
                        if dt in content:
                            rc_list.append(dt)
                    if "成立期限" in content:
                        ans = self.mrc_model.mrc("成立期限、时间", content)
                        if ans:
                            clsj_list.append(ans)
                    if "营业收入" in content:
                        ans = self.mrc_model.mrc("营业收入", content)
                        if ans:
                            yysr_list.append(ans)
                    for dt in qyxz_list:
                        if dt in content:
                            qyxz_list.append(dt)


                    # for content in self.content[i]["content"]:
                    #     if z in content:
                    #         zz_list.append(z)
                    #     if "注册" in content:
                    #         ans = self.mrc_model.mrc("企业注册地区、地址", content)
                    #         if ans:
                    #             zc_list.add(ans)
                    #     for dt in rc_data:
                    #         if dt in content:
                    #             rc_list.append(dt)
                    #     if "成立期限" in content:
                    #         ans = self.mrc_model.mrc("成立期限、时间", content)
                    #         if ans:
                    #             clsj_list.append(ans)


            target_info["企业资质"] = "，".join(zz_list)
            # print("企业注册地区")
            # print(zc_list)
            zc_regular = []
            # print("格式化-》 ", )
            # for zc_area in zc_list:
            #     reg_ind = search_area(zc_area)
            #     if reg_ind != -1:
            #         zc_regular.append(area_regular_list[reg_ind])
            #         break
            target_info["企业注册地区--文本中抽取"] = "".join(zc_list)
            target_info["企业注册地区--格式化"] = "".join(zc_regular)
            target_info["人才资质"] = "，".join(rc_list)
            target_info["成立时间"] = ",".join(clsj_list)
            target_info["企业性质"] = ",".join(qyxz_list)
            target_info["营业收入"] = ",".join(yysr_list)

            res.append(target_info)
        return res



def check_is_project_title(input_str):
    if "申报" in input_str:
        return False
    return True



class MRCMODULE(object):
    def __init__(self):
        bert_name = "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(bert_name)
        self.model = BertForQuestionAnswering.from_pretrained(bert_name)


    def mrc(self, question, text):
        # question, text = "企业资质", "（1）科技企业；（2）高新技术企业或高新技术培育企业；（3）苏州市独角兽培育企业；（4）获得各级人才计划资助的领军人才企业。"
        inputs = self.tokenizer(question, text, return_tensors='pt', return_offsets_mapping=True)
        # print(inputs)
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        # print(inputs["offset_mapping"])
        outputs = self.model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"])
        # loss = outputs.loss
        # print(outputs)
        q_len = len(question) + 2
        start_scores = torch.sigmoid(outputs.start_logits[0][q_len:])

        nd_scores = torch.sigmoid(outputs.end_logits[0][q_len:])

        start_ = []
        for i, v in enumerate(start_scores):
            if v > 0.5:
                start_.append((i, v))
        if len(start_) == 0:
            return ""
        start_.sort(key=lambda x: x[1], reverse=True)
        start = start_[0][0]

        end_ = []
        for i, v in enumerate(nd_scores):
            if v > 0.5:
                end_.append((i, v))
        if len(end_) == 0:
            return ""
        end_.sort(key=lambda x: x[1], reverse=True)
        end = end_[0][0]

        # end = torch.argmax(nd_scores)
        # # print(len(question), len(text), len(inputs["input_ids"][0]))
        # # print(start, end)
        fake_start = inputs["offset_mapping"][0][start + q_len][0]
        fake_end = inputs["offset_mapping"][0][end + q_len][1]
        return text[fake_start:fake_end]

# doc = Document()
# doc.parse_content(content2)
# doc.extract_zb()
# res = mrc("服务收入占比", "1、申报主体需获评省级服务型制造示范企业，服务收入占企业营业收入比重达30%以上；")
# print(res)

# for item in item_list:
#     file_name = "D:\data\\政策信息抽取\\{}.txt".format(item[1])
#
#     with open(file_name, "rb") as f:
#         data = f.read()
#
#     soup = BeautifulSoup(data, 'html.parser')
#     doc.parse_content(soup.text)
#     doc.extract_zb()

    # for text_row in soup.text.split("\n"):
    #     if len(text_row.strip()) == 0:
    #         continue
    #     print(text_row)
# print(search_area("苏州高新区"))
"""
    第一步分析目標
"""
