#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import re
import tika
from transformers import BertTokenizer, BertForQuestionAnswering, PreTrainedTokenizerFast
from pytorch.other_info_extraction.policy_data import content1, content2, content3, item_list

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
            "^[1-9]\..+",
            "^（[1-9]）.+",
            "^[①②③④⑤].+"
        ]
        self.mrc_model = MRCMODULE()

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


    def display_document(self):
        for span in self.content:
            print("\t"*span["title_level"] + span["title"])

    def extract_zb(self):
        condition_title_level = -1
        target_list = dict()
        for i, span in enumerate(self.content):
            title = span["title"]

            if "申报范围" in title or "申报条件" in title or "申报要求" in title or "申报对象" in title:
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
                "企业注册地区": "",
                "成立时间": "",
                "企业资质": "",
                "人才资质": "",
                "企业注册资金": ""
            }
            zz_list = []
            zc_list = set()
            rc_list = []
            clsj_list = []
            for content in self.content[target]["content"]:
                for dt in rc_data:
                    if dt in content:
                        rc_list.append(dt)

            for i, x in condition_list:
                child_list = []
                if i in self.child:
                    child_list = self.child[i]

                for z, _ in query_item_list["企业资质"].items():
                    for content in self.content[i]["content"]:
                        if z in content:
                            zz_list.append(z)
                        if "注册" in content:
                            ans = self.mrc_model.mrc("企业注册地区、地址", content)
                            if ans:
                                zc_list.add(ans)
                        for dt in rc_data:
                            if dt in content:
                                rc_list.append(dt)

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
            target_info["企业资质"] = "，".join(zz_list)
            # print("企业注册地区")
            # print(zc_list)
            zc_regular = []
            # print("格式化-》 ", )
            for zc_area in zc_list:
                reg_ind = search_area(zc_area)
                if reg_ind != -1:
                    zc_regular.append(area_regular_list[reg_ind])
                    break
            target_info["企业注册地区--文本中抽取"] = "".join(zc_list)
            target_info["企业注册地区--格式化"] = "".join(zc_regular)
            target_info["人才资质"] = "，".join(rc_list)

            res.append(target_info)
        return res




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
