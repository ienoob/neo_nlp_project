#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import sys
import re
import traceback
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            "^[0-9]{6} .+",
            "^第[一二三四五六七八九十]{1,4}条.+",
            "^第[一二三四五六七八九十]{1,4}章.+"
        ]
        # self.mrc_model = MRCMODULE()
        self.mrc_model = None

        self.root = TwoHeadTree("root")
        self.title_type2level = dict()

    def title_ex(self, input_span):
        for i, title_p in enumerate(self.title):
            if re.match(title_p, input_span):
                return "title_{}".format(i)
        return None

    def parse_content(self, content):
        try:
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
                    ind = "1"
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
                    elif title_ind == 7:
                        dt = re.search("第([一二三四五六七八九十]{1,4})条", span)
                        if dt:
                            ind = dt.group(1)
                        if "\xa0" in span:
                            span_split = [x for x in span.split("\xa0") if x]
                        elif "\u2002" in span:
                            span_split = [x for x in span.split("\u2002") if x]
                        elif " " in span:
                            span_split = [x for x in span.split(" ") if x]
                        else:
                            span_split = span.split("条")
                            span_split = ["", "条".join(span_split[1:]).strip()]
                            # print([span])
                            # raise Exception("hello")
                        span_list[i + 1]["title_content"] = span_split[1].strip()
                    elif title_ind == 8:
                        dt = re.search("第([一二三四五六七八九十]{1,4})章", span)
                        if dt:
                            ind = dt.group(1)
                        if "\xa0" in span:
                            span_split = [x for x in span.split("\xa0") if x]
                        elif "\u2002" in span:
                            span_split = [x for x in span.split("\u2002") if x]
                        elif " " in span:
                            span_split = [x for x in span.split(" ") if x]
                        else:
                            print([span])
                            raise Exception("hello")
                        span_list[i + 1]["title_content"] = span_split[1].strip()
                    else:
                        ind = "1"
                        logger.warning("can not get title index {0}, {1}".format(span, title_ind))
                        span_list[i + 1]["title_content"] = span
                    # print("st ", ind)
                    if ind in ["1", "一", "①"]:
                        if span_list[i+1]["title_type"] == 8:
                            parent[i + 1] = 0
                            child[0].append(i + 1)
                            span_list[i + 1]["title_level"] = span_list[0]["title_level"]+1
                        elif span_list[i]["title_type"] == span_list[i+1]["title_type"]:
                            parent[i + 1] = parent[i]
                            child[parent[i]].append(i + 1)
                            span_list[i + 1]["title_level"] = span_list[i]["title_level"]
                        else:
                            parent[i+1] = i
                            child[i].append(i+1)
                            span_list[i+1]["title_level"] = span_list[i]["title_level"]+1
                    else:
                        j = i
                        # if title_ind == 8:
                        #     parent[i + 1] = 0
                        #     child[0].append(i + 1)
                        #     span_list[i + 1]["title_level"] = span_list[0]["title_level"] + 1
                        #     self.title_type2level[span_list[i + 1]["title_type"]] = span_list[i + 1]["title_level"]
                        if title_ind == 7 and 8 in self.title_type2level:
                            while j >= 0:
                                if span_list[j]["title_type"] == 8:
                                    break
                                j -= 1
                            if j == -1:
                                logger.error(span)
                                logger.error(span_list)
                                logger.error("政策文本结构有些问题")
                                raise Exception("政策文本结构有些问题")
                            parent[i + 1] = j
                            child[j].append(i + 1)
                            span_list[i + 1]["title_level"] = span_list[j]["title_level"]+1
                            self.title_type2level[span_list[i + 1]["title_type"]] = span_list[i + 1]["title_level"]
                        else:
                            while j >= 0:
                                if span_list[j]["title_type"] == title_ind:
                                    break
                                j -= 1

                            if j == -1:
                                logger.error(span)
                                logger.error(span_list)
                                logger.error("政策文本结构有些问题")
                                # parent[i + 1] = i+1
                                # 忽略这个
                                # i -= 1
                                # span_list.pop()
                                # child[i+1].append(i + 1)
                                # span_list[i + 1]["title_level"] = 1
                                raise Exception
                            else:
                                parent[i + 1] = parent[j]
                                child[parent[j]].append(i+1)
                                span_list[i + 1]["title_level"] = span_list[j]["title_level"]
                                self.title_type2level[span_list[i + 1]["title_type"]] = span_list[i + 1]["title_level"]
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
        except Exception as e:
            print(span_split)
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            traceback.print_tb(exc_traceback_obj)
            logger.error(e)

    # 解析半结构化数据
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
        if len(self.content) == 0:
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
            return res
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
                    ivv = 0
                    for ci, childi in child.next.items():
                        if self.content[ci]["title_content"] in ["主体资质", "项目资质"]:
                            for cii, childii in childi.next.items():
                                conditions.append(self.content[cii]["title"])
                            ivv += 1
                    if ivv == 0:
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
                else:
                    for ci, childi in child.next.items():
                        conditions.append(self.content[ci]["title"])
                    # print(if_next_node)
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
            elif span["title_content"] == "总体申报要求":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"] in ["申报基本要求"]:
                        for cii, childii in childi.next.items():
                            conditions.append(self.content[cii]["title"])
            elif span["title_content"] == "申报材料：":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
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
                else:
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
            elif span["title_content"] == "申报范围：":
                for child_content in span["content"]:
                    conditions.append(child_content)
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
            elif span["title_content"] == "奖励程序：":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        report_processes.append(self.content[ci]["title"])
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
            elif span["title_content"] == "资助类别和标准":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        project_names.append(sub_titile_content)
            elif span["title_content"] == "申报地区总部需符合下列条件：":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    project_info = {
                        "project_name": "申报地区总部",
                        "text_materials": [],
                        "conditions": [],
                        "apply_dates": []
                    }
                    sub_condition = []
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        sub_condition.append(sub_titile_content)
                    project_info["conditions"] = sub_condition
                    project_infos.append(project_info)
            elif span["title_content"] == "申报总部型机构需符合下列条件：":
                if_next_node = 0
                for ci, childi in child.next.items():
                    if len(childi.next):
                        if_next_node += 1
                if len(child.next) == 0:
                    pass
                elif if_next_node == 0:
                    project_info = {
                        "project_name": "申报总部型机构",
                        "text_materials": [],
                        "conditions": [],
                        "apply_dates": []
                    }
                    sub_condition = []
                    for ci, childi in child.next.items():
                        sub_titile_content = self.content[ci]["title_content"]
                        sub_condition.append(sub_titile_content)
                    project_info["conditions"] = sub_condition
                    project_infos.append(project_info)
            elif span["title_content"] == "指导用好普惠性纾困政策":
                # print(span)
                for ci, childi in child.next.items():
                    project_info = {
                        "project_name": self.content[ci]["title_content"],
                        "text_materials": [],
                        "conditions": [],
                    }
                    for cii, childii in childi.next.items():
                        project_info["conditions"].append(self.content[cii]["title_content"])
                    project_infos.append(project_info)
            elif span["title_content"] == "适用范围":
                for sub_content in self.content[c]["content"]:
                    conditions.append(sub_content)
            elif span["title_content"] == "补贴标准":
                for sub_content in self.content[c]["content"]:
                    benefit_infos.append(sub_content)
            elif span["title_content"] == "申请条件":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"][-7:] == "满足以下条件：":
                        project_name = re.search("申请([\u4e00-\u9fa5]+?服务券)", self.content[ci]["title_content"])
                        if not project_name:
                            project_name = re.search("接受([\u4e00-\u9fa5]+?服务券)", self.content[ci]["title_content"])
                        if project_name:
                            project_name = project_name.group(1)
                            project_info = {
                                "project_name": project_name,
                                "text_materials": [],
                                "conditions": [],
                            }
                            for cii, childii in childi.next.items():
                                project_info["conditions"].append(self.content[cii]["title_content"])
                            project_infos.append(project_info)
            elif span["title_content"][:4] == "申请材料":
                if span["title_content"][-7:] == "提供以下材料：":
                    for ci, childi in child.next.items():
                        text_materials.append(self.content[ci]["title_content"])
            elif span["title_content"][:10] == "申请条件及奖励标准。":
                conditions.append(span["title_content"][10:])
                benefit_infos.append(span["title_content"][10:])
            elif span["title_content"][:5] == "本办法所指":
                conditions.append(span["title_content"])
            elif span["title_content"] == "支持范围和补助标准":

                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"] == "补助标准：":
                        benefit_infos = self.content[ci]["content"]
            elif span["title_content"] == "申报流程和时间安排":
                for ci, childi in child.next.items():
                    report_processes.append(self.content[ci]["title_content"])
            elif span["title_content"] == "有关要求":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"] == "申报项目查重要求":
                        for cii, childii in childi.next.items():
                            conditions.append(self.content[cii]["title_content"])
                    elif self.content[ci]["title_content"] == "申报项目资金要求":
                        for sub_content in self.content[ci]["content"]:
                            conditions.append(sub_content)
            elif span["title_content"] == "申报书附件内容":
                for ci, childi in child.next.items():
                    text_materials.append(self.content[ci]["title_content"])
            elif span["title_content"] == "补助对象、原则和标准":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"] == "补助对象和条件":
                        for cii, childii in childi.next.items():
                            conditions.append(self.content[cii]["title"])
                    elif self.content[ci]["title_content"] == "补助标准":
                        for cii, childii in childi.next.items():
                            if self.content[cii]["title_content"] == "基础补助额":
                                for ciii, childiii in childii.next.items():
                                    benefit_infos.append("{}-{}".format("基础补助额", self.content[ciii]["title_content"]))
                            elif self.content[cii]["title_content"] == "增量补助额":
                                for ciii, childiii in childii.next.items():
                                    benefit_infos.append("{}-{}".format("增量补助额", self.content[ciii]["title_content"]))
                            elif self.content[cii]["title_content"] == "最终补助额":
                                for sub_content in self.content[cii]["content"]:
                                    benefit_infos.append("{}-{}".format("最终补助额", sub_content))
            elif span["title_content"] == "申请项目资金应符合下列基本条件：":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"][-6:] == "项目的条件：":
                        project_name = re.search("申请([\u4e00-\u9fa5]+?)的条件", self.content[ci]["title_content"])
                        if project_name:
                            project_name = project_name.group(1)
                            project_info = {
                                "project_name": project_name,
                                "text_materials": [],
                                "conditions": [],
                            }
                            for cii, childii in childi.next.items():
                                project_info["conditions"].append(self.content[cii]["title_content"])
                            project_infos.append(project_info)
            elif span["title_content"] == "资助标准和申请条件":
                for ci, childi in child.next.items():
                    if "普通高中国家助学金用于资助家庭经济困难学生的学习和生活费用开支" in self.content[ci]["title_content"]:
                        benefit_infos.append(self.content[ci]["title_content"])
            elif span["title_content"] == "支持企业稳定发展":
                # print(child.next)
                for ci, childi in child.next.items():
                    # print(self.content[ci]["title_content"])
                    if re.search("可申请最高\d+?万元的创业担保贷款", self.content[ci]["title_content"]):
                        benefit_infos.append(self.content[ci]["title_content"])
            elif span["title_content"] == "支持范围":
                for ci, childi in child.next.items():
                    if self.content[ci]["title_content"][-5:] == "支持范围：":
                        project_info = {
                            "project_name": "",
                            "text_materials": [],
                            "conditions": [],
                        }
                        for cii, childii in childi.next.items():
                            project_info["conditions"].append(self.content[cii]["title_content"])
                        project_infos.append(project_info)

        # 标题子目录问题
        if len(conditions) == 0 and len(project_names) == 0 and len(text_materials) == 0 and len(report_processes) == 0 and len(project_infos)==0:
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

        if len(conditions) == 0 and len(project_names) == 0 and len(text_materials) == 0 and len(report_processes) == 0 and len(project_infos)==0:
            res = self.parse_half_struction_v2()
            project_infos = res["project_infos"]

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

    def parse_half_struction_v2(self):
        project_infos = []

        for c, child in self.root.next.items():
            # print(self.content[c]["title_content"])
            for ci, childi in child.next.items():
                if self.content[ci]["title_content"][:6] == "符合以下条件":
                    project_info = {
                        "project_name": self.content[c]["title_content"],
                        "conditions": []
                    }
                    for cii, childii in childi.next.items():
                        project_info["conditions"].append(self.content[cii]["title_content"])
                    project_infos.append(project_info)
                elif self.content[ci]["title_content"][-8:] == "需满足以下条件：":
                    project_info = {
                        "project_name": self.content[c]["title_content"],
                        "conditions": []
                    }
                    for cii, childii in childi.next.items():
                        project_info["conditions"].append(self.content[cii]["title_content"])
                    project_infos.append(project_info)
                elif self.content[ci]["title_content"][-7:] == "满足以下条件：":
                    project_info = {
                        "project_name": self.content[c]["title_content"],
                        "conditions": []
                    }
                    for cii, childii in childi.next.items():
                        project_info["conditions"].append(self.content[cii]["title_content"])
                    project_infos.append(project_info)
                elif self.content[ci]["title_content"][-7:] == "提供以下材料：":
                    project_info = {
                        "project_name": self.content[c]["title_content"],
                        "text_materials": []
                    }
                    for cii, childii in childi.next.items():
                        project_info["text_materials"].append(self.content[cii]["title_content"])
                    project_infos.append(project_info)
                elif self.content[ci]["title_content"][:6] == "奖励金额按照":
                    project_info = {
                        "project_name": self.content[c]["title_content"],
                        "benefit_infos": [self.content[ci]["title_content"][4:]]
                    }
                    project_infos.append(project_info)
                elif self.content[ci]["title_content"][:7] == "课题资助标准为":
                    project_info = {
                        "project_name": self.content[c]["title_content"],
                        "benefit_infos": [self.content[ci]["title_content"][7:]]
                    }
                    project_infos.append(project_info)
        project_info_out = {}
        # 融合结果
        for project_info in project_infos:
            if project_info["project_name"] in project_info_out:
                for k, v in project_info.items():
                    if k == "project_name":
                        continue
                    project_info_out[project_info["project_name"]][k] = v
            else:
                project_info_out[project_info["project_name"]] = project_info
        project_infos_out = [v for _, v in project_info_out.items()]

        return {
            "project_infos": project_infos_out
        }

    def parse_example(self):
        res = {
            "zhibiao": [
                ("企业注册地区", "属于", "上海", ""),
                ("企业注册日期", "小于等于", "2021年12月2日", ""),
                ("注册资本", "大于", "1", "元"),
                ("企业性质", "属于", "科技服务机构", ""),
                ("企业类型", "属于", "国资", ""),
                ("企业资质", "属于", "高新技术企业", ""),
                ("企业技术能力", "属于", "工业设计中心", ""),
                ("人才资质", "属于", "领军人物", ""),
                ("企业行业", "属于", "武术行业", ""),
                ("企业人数", "大于等于", "1", "人"),
                ("领军人数", "大于等于", "1", "人"),
                ("大专及以上人数", "大于等于", "1", "人"),
                ("本科及以上人数", "大于等于", "1", "人"),
                ("硕士及以上人数", "大于等于", "1", "人"),
                ("博士及以上人数", "大于等于", "1", "人"),
                ("企业总资产", "大于等于", "1", "日元"),
                ("企业实收资本", "大于等于", "1", "津巴布韦币"),
                ("净资产总额", "大于等于", "1", "津巴布韦币"),
                ("净利润", "大于等于", "1", "美元"),
                ("营业收入", "大于等于", "1", "美元"),
                ("税收", "大于等于", "1", "人民币"),
                ("知识产权总数", "大于等于", "1", "人"),
                ("专利总数", "大于等于", "1", "人"),
                ("发明专利", "大于等于", "1", "人"),
                ("注册商标总数", "大于等于", "1", "人"),
                ("软件著作权总数", "大于等于", "1", "人"),
                ("软件著作权", "大于等于", "1", "人"),
                ("研究开发投入", "大于等于", "1万亿", "美元"),
                ("研发人数", "大于等于", "1万", "人")
            ],
            "zhibiao_person": []
        }
        return res

    def parse_precision_v2(self, conditions):
        res = {
            "mm": []
        }
        for condition in conditions:
            if "具有独立法人资格" in condition:
                res["mm"].append(("具有独立法人资格", condition))
            if "注册" in condition:
                rs = re.search("(注册地在[\u4e00-\u9fa5]范围内)", condition)
                if rs:
                    res["mm"].append((rs.group(1), condition))
            if "持续经营" in condition:
                res["mm"].append(("持续经营", condition))
            if "税务关系隶属于" in condition:
                rs = re.search("(税务关系隶属于[\u4e00-\u9fa5])", condition)
                if rs:
                    res["mm"].append((rs.group(1), condition))
            if "业务经营正常" in condition:
                res["mm"].append(("业务经营正常", condition))
            if "无严重失信行为" in condition:
                rs = re.search("(近[一二三四五六七]年无严重失信行为)", condition)
                if rs:
                    res["mm"].append((rs.group(1), condition))

    # 精细解析
    def parse_precision(self, conditions):
        res = {
            "zhibiao": [],
            "zhibiao_person": [],
            "micro_condition": [],
            "for_train": []
        }
        for condition in conditions:
            if "注册" in condition:
                rs = re.search("在([\u4e00-\u9fa5]{1,10}?)依法登记注册", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                else:
                    rs = re.search("在([\u4e00-\u9fa5]{1,10}?)注册", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("注册地在([\u4e00-\u9fa5]+?)范围内", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                        res["for_train"].append({"line": condition, "span": rs.group(1), "index": rs.span(1)})

                    rs = re.search("申报单位应在([\u4e00-\u9fa5]+?)设立", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("登记注册在([\u4e00-\u9fa5]+?);", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("注册在([\u4e00-\u9fa5]+?)内", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("注册地在([\u4e00-\u9fa5]+?)的企业（(.+?)）", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                        # res["zhibiao"].append(("企业注册地区", "小于等于", rs.group(2)))
                    rs = re.search("注册([0-9]+?年)以上", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(1), ""))
                    rs = re.search("注册登记、税务登记和主要工作场所均在([\u4e00-\u9fa5]+)", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("为([\u4e00-\u9fa5]+)登记注册", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("注册地在([\u4e00-\u9fa5]+)的企业", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("注册登记.税务登记和主要工作场所均在([\u4e00-\u9fa5]+)", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("正常经营([0-9]+年以上)", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(1), ""))
                    rs = re.search("工商注册地、税务征管关系及统计关系在([\u4e00-\u9fa5]+)范围内", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search("注册成立([一二三四五六七八九十])年（365个日历天数）以上", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(1), "年"))
                    rs = re.search(r"(\d{1,4}年\d{1,2}月\d{1,2}日)前注册", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册日期", "小于等于", rs.group(1), ""))
                    rs = re.search(r"注册地址在(.+?)的", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search(r"注册经营满(\d+?)年", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(1), "年"))
                    rs = re.search(r"申请者([\u4e00-\u9fa5]+)工商注册并经营满([一二三四五六七八九十])年", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(2), "年"))
                    rs = re.search(r"申请者([\u4e00-\u9fa5]+)工商注册并经营满([一二三四五六七八九十])年", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(2), "年"))
                    rs = re.search(r"市、区两级认定的([\u4e00-\u9fa5]+?)内注册", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search(r"注册登记([一二三四五六七八九十]+?)年以上", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(1), "年"))
                    rs = re.search(r"注册地在([\u4e00-\u9fa5]+?)且", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search(r"\.([\u4e00-\u9fa5]+?)内注册", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    rs = re.search(r"）([\u4e00-\u9fa5]+?)内注册", condition)
                    if rs:
                        res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
            if "成立" in condition:
                rs = re.search("在(.+?)成立([一二三四五六])年以上", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))
                    res["zhibiao"].append(("企业注册年限", "大于等于", rs.group(2), "年"))
                rs = re.search("企业成立([一二三四五六七八九十])年以内", condition)
                if rs:
                    res["zhibiao"].append(("企业注册年限", "小于等于", rs.group(1), "年"))
            if "设立" in condition:
                rs = re.search("在([\u4e00-\u9fa5]+)依法设立", condition)
                if rs:
                    res["zhibiao"].append(("企业注册地区", "属于", rs.group(1), ""))

            if "营业收入" in condition:
                rs = re.search("营业收入([0-9]+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("营业收入", "大于等于", rs.group(1), "元"))
                else:
                    rs = re.search("营业收入总额在([0-9]+?万)元以上", condition)
                    if rs:
                        res["zhibiao"].append(("营业收入", "大于等于", rs.group(1), "元"))
            if "利税" in condition:
                rs = re.search("利税总额([0-9]+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("税收", "大于等于", rs.group(1), "元"))
            if "三年营业收入平均增长率" in condition:
                rs = re.search("三年营业收入平均增长率([0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("三年营业收入平均增长率", "大于等于", rs.group(1), "百分比"))
            if "平均营业利润率" in condition:
                rs = re.search("平均营业利润率([0-9]+?%)", condition)
                if rs:
                    res["zhibiao"].append(("平均营业利润率", "大于等于", rs.group(1), "百分比"))
            if "发明专利" in condition:
                rs = re.search("拥有([0-9]+?)项及以上发明专利", condition)
                if rs:
                    res["zhibiao"].append(("发明专利", "大于等于", rs.group(1), "项"))
            if "专利总数" in condition:
                rs = re.search("专利总数(不少于[0-9]+?)项", condition)
                if rs:
                    res["zhibiao"].append(("发明专利", "大于等于", rs.group(1), "项"))
            if "软件著作权" in condition:
                rs = re.search("软件著作权[\u4e00-\u9fa5、]+不少于([0-9]+?)项", condition)
                if rs:
                    res["zhibiao"].append(("软件著作权", "大于等于", rs.group(1), "项"))
            if "软件产品评估" in condition:
                rs = re.search("软件产品评估[\u4e00-\u9fa5、]+不少于([0-9]+?)项", condition)
                if rs:
                    res["zhibiao"].append(("软件产品评估", "大于等于", rs.group(1), "项"))
            if "技术中心工作人员" in condition:
                rs = re.search("技术中心工作人员数不低于([0-9]+?)人", condition)
                if rs:
                    res["zhibiao"].append(("技术中心工作人员", "大于等于", rs.group(1), "人"))
            if "年主营业务收入" in condition:
                rs = re.search("年主营业务收入不低于([0-9\s]+?亿)元", condition)
                if rs:
                    res["zhibiao"].append(("营业收入", "大于等于", rs.group(1), "元"))
            if "研究开发投入" in condition:
                rs = re.search("支出额不低于([0-9\s]+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("研究开发投入", "大于等于", rs.group(1), "元"))
            if "技术带头人" in condition:
                rs = re.search("人员数不少于([0-9]+?)人", condition)
                if rs:
                    res["zhibiao"].append(("技术中心工作人员", "大于等于", rs.group(1), "人"))

            if "企业实收资本" in condition:
                rs = re.search("企业实收资本(不低于[0-9]+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("企业实收资本", "大于等于", rs.group(1), "元"))
            if "研发费用总额占同期销售收入总额的比例" in condition:
                if "比例符合以下标准：" in condition:
                    zb = []
                    rs = re.findall("销售收入为([0-9亿万—以下上元]+)的?企业，比例(不低于[0-9]+?)％", condition)
                    # for r in rs:
                    #     zb.append({"销售收入": r[0], "比例": r[1]})
                    # if zb:
                    #     res["zhibiao"].append(("研发费用总额占同期销售收入总额的比例", zb))
            if "研发投入占营业收入比例" in condition:
                rs = re.search("研发投入占营业收入比例(不低于[0-9]+?)%", condition)
                if rs:
                    res["zhibiao"].append(("研发投入占营业收入比例", "大于等于", rs.group(1), "百分比"))
            if "高新技术企业" in condition:
                res["zhibiao"].append(("企业资质", "属于", "高新技术企业", ""))
            if "尚未在主板.创业板.科创板上市或在“新三板”挂牌" in condition:
                res["zhibiao"].append(("企业资质", "属于", "未上市", ""))
            if "失信" in condition:
                rs = re.search("近([0-9]+?)年无严重失信行为", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", "大于等于", rs.group(1), "年"))
                rs = re.search("近([一二两三四五六七八九])年内?无严重失信行为", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", "大于等于", rs.group(1), "年"))
                rs = re.search("近([0-9]+?年)信用状况良好，无严重失信行为", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", "大于等于", rs.group(1)))
                rs = re.search("近([一二两三四五六七八九])年内?信用状况良好，无重大失信记录", condition)
                if rs:
                    res["zhibiao"].append(("无失信行为", "大于等于", rs.group(1), "年"))
            if "年度研究与试验发展经费" in condition:
                rs = re.search("年度研究与试验发展经费支出额不低于([0-9]+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("研究开发投入", "大于等于", rs.group(1), "元"))

            if "研究与试验发展人员" in condition:
                rs = re.search("研究与试验发展人员数不低于([0-9\s]+?)人", condition)
                if rs:
                    res["zhibiao"].append(("研发人数", "大于等于", rs.group(1), "人"))
            if "上市公司" in condition:
                res["zhibiao"].append(("企业资质", "属于", "上市公司", ""))
            if "农机（农业）企业" in condition:
                res["zhibiao"].append(("企业行业", "属于", "农业", ""))
            if "属于科技型企业" in condition:
                res["zhibiao"].append(("企业性质", "属于", "科技", ""))
            if "净资产总额" in condition:
                rs = re.search("净资产总额不超过([0-9]+?[万亿])元", condition)
                if rs:
                    res["zhibiao"].append(("净资产总额", "大于等于", rs.group(1), "元"))
            if "销售总额" in condition:
                rs = re.search("销售总额不超过([0-9]+?[万亿]元)", condition)
                if rs:
                    res["zhibiao"].append(("营业收入", "大于等于", rs.group(1), "元"))
            if "拥有对外贸易经营" in condition:
                res["zhibiao"].append(("企业行业", "属于", "对外贸易", ""))
            if "从事文化和旅游装备技术研发" in condition:
                res["zhibiao"].append(("企业行业", "属于", "文化和旅游", ""))
            if "核定为质量信用" in condition:
                rs = re.findall("核定为质量信用([ABC]+?级等级)满([0-9]?)年以上", condition)
                for sub in rs:
                    res["zhibiao"].append(("企业资质", "属于", sub, ""))
            if "工业企业质量信用评价" in condition:
                res["zhibiao"].append(("企业行业", "属于", "工业、制造业", ""))
            if "农业" in condition:
                res["zhibiao"].append(("企业行业", "属于", "农业", ""))
            if "具有知识产权" in condition:
                res["zhibiao"].append(("知识产权总数", "大于等于", "1", "个"))
            if "知识产权运营服务机构持有可运营专利" in condition:
                res["zhibiao"].append(("知识产权总数", "大于等于", "1", "个"))
                res["zhibiao"].append(("专利总数", "大于等于", "1", "个"))
            # if "金融机构" in condition:
            #     res["zhibiao"].append(("企业行业", "属于", "金融", ""))
            if "拥有自主有效的国内注册商标" in condition:
                res["zhibiao"].append(("注册商标总数", "大于等于", "1", "个"))
            if "外资总部型企业" in condition:
                res["zhibiao"].append(("企业性质", "属于", "外资总部型", ""))
            if "省级跨国公司地区总部" in condition:
                res["zhibiao"].append(("企业性质", "属于", "跨国公司地区总部", ""))
            tp = re.search("详见(《[\S]+?》)", condition)
            if tp:
                res["zhibiao"].append(("链接资料", "", tp.group(1), ""))
            if "“独角兽”培育企业库" in condition:
                res["zhibiao"].append(("企业资质", "属于", "独角兽", ""))
            if "人工智能场景应用重点领域" in condition:
                res["zhibiao"].append(("企业行业", "属于", "人工智能", ""))
            tp = re.search("申报项目必须符合(.+?)（", condition)
            if tp:
                res["zhibiao"].append(("链接资料", "", tp.group(1), ""))
            tp = re.findall("满足(《[\u4e00-\u9fa5]+?》)和(《[\u4e00-\u9fa5]+?》)中明确的具体条件", condition)
            for sut in tp:
                res["zhibiao"].append(("链接资料", "", sut, ""))
            tp = re.findall("满足(《[\u4e00-\u9fa5]+?》)中明确的具体条件", condition)
            for sut in tp:
                res["zhibiao"].append(("链接资料", "", sut, ""))
            if "学位" in condition:
                rs = re.search("具有(.+?)学位", condition)
                if rs:
                    res["zhibiao_person"].append(("学位", "等于", rs.group(1), ""))
                rs = re.search("拥有(.+?)学位", condition)
                if rs:
                    res["zhibiao_person"].append(("学位", "等于", rs.group(1), ""))
            if "股权" in condition:
                rs = re.search("股权一般不低于([0-9]+)%", condition)
                if rs:
                    res["zhibiao_person"].append(("股权", "大于等于", rs.group(1), "百分比"))
            if "现金出资" in condition:
                rs = re.search("现金出资（实收资本，不含技术入股）不少于([0-9]+万)元", condition)
                if rs:
                    res["zhibiao_person"].append(("现金出资", "大于等于", rs.group(1), "元"))
            if "年龄" in condition:
                rs = re.search("年龄一般(不超过[0-9]+?)周岁", condition)
                if rs:
                    res["zhibiao_person"].append(("年龄",  "小于等于", rs.group(1), "周岁"))
                rs = re.search("年龄(不超过[0-9]+?)周岁", condition)
                if rs:
                    res["zhibiao_person"].append(("年龄",  "小于等于", rs.group(1), "周岁"))
            if "引才企业应具备以下条件" in condition:
                if "国家高新技术企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "属于", "国家高新技术企业", ""))
                if "市级高成长性创新型企业培育计划" in condition:
                    res["zhibiao_person"].append(("企业资质", "属于", "市级高成长性创新型企业培育计划", ""))
                if "瞪羚企业培育计划入选企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "属于", "瞪羚企业培育计划入选企业", ""))
                if "市级以上领军人才创办的企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "属于", "市级以上领军人才创办的企业", ""))
                if "市级及以上研发机构的企业" in condition:
                    res["zhibiao_person"].append(("企业资质", "属于", "市级及以上研发机构的企业", ""))
            tp = re.search("(《.+?》)明确的具体申报条件", condition)
            if tp:
                res["zhibiao"].append(("链接资料", "", tp.group(1), ""))
            tp = re.search("(《.+?》)规定的具体条件", condition)
            if tp:
                res["zhibiao"].append(("链接资料", "", tp.group(1), ""))
            tp = re.search("(《.+?》)明确的具体条件", condition)
            if tp:
                res["zhibiao"].append(("链接资料", "", tp.group(1), ""))
            if "研究开发费用占销售收入的比重" in condition:
                rs = re.search("研究开发费用占销售收入的比重不低于([0-9]+?)%", condition)
                if rs:
                    res["zhibiao"].append(("研究开发费用占销售收入的比重", "大于等于", rs.group(1), "百分比"))
            if "中小微企业" in condition:
                res["zhibiao"].append(("企业资质", "属于", "中小微企业", ""))
            if "研发经费支出占主营业务收入" in condition:
                rs = re.search("研发经费支出占主营业务收入的比重应高于([0-9]+)%", condition)
                if rs:
                    res["zhibiao"].append(("研发经费支出占主营业务收入", "大于", rs.group(1), "百分比"))
            if "企业未上市" in condition:
                res["zhibiao"].append(("企业资质", "属于", "未上市", ""))
            if "入选苏州市瞪羚计划企业" in condition:
                res["zhibiao"].append(("企业资质", "属于", "瞪羚计划企业", ""))
            if "电子商务年交易额" in condition:
                rs = re.search("电子商务年交易额达到([0-9]+?亿)元以上", condition)
                if rs:
                    res["zhibiao"].append(("电子商务年交易额", "大于", rs.group(1), "元"))
            if "电子商务服务业年营业收入" in condition:
                rs = re.search("电子商务服务业年营业收入达到([0-9]+?万)元以上", condition)
                if rs:
                    res["zhibiao"].append(("电子商务服务业年营业收入", "大于", rs.group(1), "元"))
            if "投入运营" in condition:
                rs = re.search("正式投入运营([0-9]+?年)以上", condition)
                if rs:
                    res["zhibiao"].append(("运营时间", "大于", rs.group(1), "年"))
            if "国内外知名企业、高校、科研单位从事研发及管理经历" in condition:
                rs = re.search("有(.+?)年国内外知名企业、高校、科研单位从事研发及管理经历", condition)
                if rs:
                    res["zhibiao_person"].append(("国内外知名企业、高校、科研单位从事研发及管理经历", "大于等于", rs.group(1), "年"))
            if "昆山市级及以上人才计划认定" in condition:
                res["zhibiao_person"].append(("昆山市级及以上人才计划", "等于", "认定", ""))
            if "项目总投入" in condition:
                rs = re.search("项目总投入不少于([.+?]万)元", condition)
                if rs:
                    res["zhibiao_person"].append(("项目总投入", "大于等于",rs.group(1), "元"))
            if "研发、管理工作经历或自主创业经历" in condition:
                rs = re.search("具有([0-9]+?)年以上相关研发、管理工作经历或自主创业经历", condition)
                if rs:
                    res["zhibiao_person"].append(("相关研发、管理工作经历或自主创业经历", "大于", rs.group(1), "年"))
            if "具有自主知识产权" in condition:
                res["zhibiao_person"].append(("知识产权总数", "大于等于", "1", "个"))
            if "税前年薪" in condition:
                rs = re.search("税前年薪不低于([0-9]+?万)元", condition)
                if rs:
                    res["zhibiao_person"].append(("税前年薪", "大于等于", rs.group(1), "元"))
            if "苏州市瞪羚计划企业" in condition:
                res["zhibiao_person"].append(("企业资质", "属于", "苏州市瞪羚计划企业", ""))
            if "苏州市“独角兽”培育企业" in condition:
                res["zhibiao_person"].append(("企业资质", "属于", "苏州市“独角兽”培育企业", ""))
            if "项目注册资本" in condition:
                rss = re.findall("项目注册资本不低于([0-9]+?万)元", condition)
                for rs in rss:
                    res["zhibiao"].append(("项目注册资本", "大于等于", rs, "元"))
            if "实缴出资额" in condition:
                rs = re.search("实缴出资额(不少于[0-9]+?万)元", condition)
                if rs:
                    res["zhibiao_person"].append(("实缴出资额", "大于等于", rs.group(1), "元"))
            if "内容详见" in condition:
                rs = re.search("内容详见(.+?)。", condition)
                if rs:
                    res["zhibiao_person"].append(("链接文件", "", rs.group(1), ""))
            if "注册资金" in condition:
                rs = re.search("注册资金在([0-9]+?万)元人民币以上", condition)
                if rs:
                    res["zhibiao"].append(("注册资本", "大于", rs.group(1), "元人民币"))
            if "职工人数" in condition:
                rs = re.search("职工人数一般在([0-9]+?)人以上", condition)
                if rs:
                    res["zhibiao"].append(("企业人数", "大于", rs.group(1), "人"))
            if "生产经营" in condition:
                rs = re.search("生产经营([0-9]+?)年以上", condition)
                if rs:
                    res["zhibiao"].append(("经营时间", "大于", rs.group(1), "年"))
            if "企业成立日期" in condition:
                rs = re.search("企业成立日期在([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日)（含）以后", condition)
                if rs:
                    res["zhibiao"].append(("企业成立日期", "大于等于", rs.group(1), ""))
                rs = re.search("企业成立日期在([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日（含）-[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日（含）之间)", condition)
                if rs:
                    res["zhibiao"].append(("企业成立日期", "大于等于", rs.group(1), ""))
            if "最后一轮投后估值" in condition:
                rs = re.search("最后一轮投后估值([0-9]+?)，", condition)
                if rs:
                    res["zhibiao"].append(("最后一轮投后估值", "大于等于", rs.group(1), "元"))
            if "最新一轮投后估值" in condition:
                rs = re.search("最新一轮投后估值([0-9\-]+亿美元或[0-9\-]+亿元人民币)", condition)
                # if rs:
                #     res["zhibiao"].append(("最新一轮投后估值", "大于等于", rs.group(1), ""))
            if "未在主板或创业板上市" in condition:
                res["zhibiao"].append(("企业资质", "属于", "未上市", ""))
            if "上年度销售收入" in condition:
                rs = re.search("企业上年度销售收入（不含流水.关联交易）不低于([0-9]+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("上年度销售收入", "大于等于", rs.group(1), "元"))
            if "三年销售收入" in condition and "净利润平均增长率" in condition:
                rs = re.search("三年销售收入（不含流水.关联交易）或净利润平均增长率不低于([0-9]+?)%", condition)
                if rs:
                    res["zhibiao"].append(("三年销售收入增长率", "大于等于", rs.group(1), "百分比"))
                    res["zhibiao"].append(("净利润平均增长率", "大于等于", rs.group(1), "百分比"))
            if "业务收入" in condition:
                rs = re.search("([0-9]{4}年度相关业务收入)应不低于([0-9]+?万)元", condition)
                if rs:
                    res["zhibiao"].append((rs.group(1), "大于等于", rs.group(2), "元"))
            if "研发人员" in condition:
                rs = re.search("研发人员原则上(\d+)人以上", condition)
                if rs:
                    res["zhibiao"].append(("研发人数", "大于等于", rs.group(1), "人"))
            if "先进制造业和服务业企业" in condition:
                res["zhibiao"].append(("企业行业", "属于", "制造业和服务业", ""))
            if "购置设备的发票金额" in condition:
                rs = re.search("购置设备的发票金额需达到(\d+万元)", condition)
                if rs:
                    res["zhibiao"].append(("购置设备的发票金额", "大于等于", rs.group(1), ""))
            if "境外母公司资产总额" in condition:
                rs = re.search("境外母公司资产总额不低于(\d+?亿)美元", condition)
                if rs:
                    res["zhibiao"].append(("境外母公司资产总额", "大于等于", rs.group(1), "美元"))
            if "境外母公司持股比例" in condition:
                rs = re.search("境外母公司持股比例不低于(\d+?)%", condition)
                if rs:
                    res["zhibiao"].append(("境外母公司持股比例", "大于等于", rs.group(1), "百分比"))
            if "投资的境内外独立法人企业" in condition:
                rs = re.search("投资的境内外独立法人企业不少于(\d+?)家", condition)
                if rs:
                    res["zhibiao"].append(("投资的境内外独立法人企业", "大于等于", rs.group(1), "家"))
            if "设立的境内外分支机构" in condition:
                rs = re.search("设立的境内外分支机构不少于(\d+?)家", condition)
                if rs:
                    res["zhibiao"].append(("设立的境内外分支机构", "大于等于", rs.group(1), "家"))
            if "授权管理（服务）的境内外独立法人企业" in condition:
                rs = re.search("授权管理（服务）的境内外独立法人企业不少于(\d+?)家", condition)
                if rs:
                    res["zhibiao"].append(("授权管理（服务）的境内外独立法人企业", "大于等于", rs.group(1), "家"))
            if "授权管理（服务）的境内外分支机构" in condition:
                rs = re.search("授权管理（服务）的境内外分支机构不少于(\d+?)家", condition)
                if rs:
                    res["zhibiao"].append(("授权管理（服务）的境内外分支机构", "大于等于", rs.group(1), "家"))
            if "建筑面积" in condition:
                rs = re.search("建筑面积不少于(\d+?)平方米", condition)
                if rs:
                    res["zhibiao"].append(("建筑面积", "大于等于", rs.group(1), "平方米"))
            if "人均使用面积" in condition:
                rs = re.search("人均使用面积不少于(\d+?)平方米", condition)
                if rs:
                    res["zhibiao"].append(("人均使用面积", "大于等于", rs.group(1), "平方米"))
            if "创业服务工作的专职人员" in condition:
                rs = re.search("配备(\d+?)名以上从事创业服务工作的专职人员", condition)
                if rs:
                    res["zhibiao"].append(("创业服务工作的专职人员", "大于等于", rs.group(1), "名"))
            if "生产经营满6个月在孵企业" in condition:
                rs = re.search("生产经营满6个月在孵企业不少于(\d+?)户", condition)
                if rs:
                    res["zhibiao"].append(("生产经营满6个月在孵企业", "大于等于", rs.group(1), "户"))
            if "平均每户带动就业" in condition:
                rs = re.search("平均每户带动就业不少于(\d+?)人", condition)
                if rs:
                    res["zhibiao"].append(("平均每户带动就业", "大于等于", rs.group(1), "人"))
            if "月销售额" in condition:
                rs = re.search("月销售额(\d+?万)元以下（含本数）的旅行社", condition)
                if rs:
                    res["zhibiao"].append(("月销售额", "小于等于", rs.group(1), "元"))
                    res["zhibiao"].append(("所属领域", "属于", "旅游业", ""))
            if re.search("申请者为[\u4e00-\u9fa5]+?科研院所", condition):
                res["zhibiao"].append(("企业类型", "属于", "科研院所", ""))
            if "回国时间" in condition:
                rs = re.search("回国时间不超过(\d+)年", condition)
                if rs:
                    res["zhibiao"].append(("回国时间", "小于等于", rs.group(1), "年"))
            if "近三年研发总投入占销售总收入" in condition:
                rs = re.search("近三年研发总投入占销售总收入比例达(\d+?)%以上", condition)
                if rs:
                    res["zhibiao"].append(("近三年研发总投入占销售总收入", "大于等于", rs.group(1), "百分比"))
            if "上一年年销售收入" in condition:
                rs = re.search("上一年年销售收入不低于(\d+?万)元", condition)
                if rs:
                    res["zhibiao"].append(("上年度销售收入", "大于等于", rs.group(1), "元"))
            if "认定国家高企" in condition:
                res["zhibiao"].append(("企业资质", "属于", "高新技术企业", ""))
            if "辅导企业申报国家高企的成功案例" in condition:
                rs = re.search("辅导企业申报国家高企的成功案例，至少(\d+?)例", condition)
                if rs:
                    res["zhibiao"].append(("辅导企业申报国家高企的成功案例", "大于等于", rs.group(1), "例"))
            if "独立法人资格的生物医药企业或机构" in condition:
                res["zhibiao"].append(("所属领域", "属于", "生物医药", ""))
            if "瞪羚企业" in condition:
                res["zhibiao"].append(("企业资质", "属于", "瞪羚企业", ""))
            if "科技领军企业和领军培育企业" in condition:
                res["zhibiao"].append(("企业资质", "属于", "科技领军企业和领军培育企业", ""))
            if "纳入国家科技统计调查的企业" in condition:
                res["zhibiao"].append(("企业资质", "属于", "纳入国家科技统计调查的企业", ""))
            if "项目负责人同期主持市级各类科技计划项目数" in condition:
                rs = re.search("项目负责人同期主持市级各类科技计划项目数不得超过(\d+?)项", condition)
                if rs:
                    res["zhibiao"].append(("项目负责人同期主持市级各类科技计划项目数", "小于等于", rs.group(1), "项"))
            if "项目申请财政资金额不超过项目总资金" in condition:
                rs = re.search("项目申请财政资金额不超过项目总资金的(\d+)%", condition)
                if rs:
                    res["zhibiao"].append(("项目申请财政资金额占项目总资金比例", "小于等于", rs.group(1), "百分比"))
            if "未被列入失信行为记录" in condition:
                res["zhibiao"].append(("失信行为记录", "等于", "0", ""))
            if "上年度海关进出口额" in condition:
                rs = re.search("上年度海关进出口额在(\d+?万)美元（含\d+万美元）以下", condition)
                if rs:
                    res["zhibiao"].append(("上年度海关进出口额", "大于等于", rs.group(1), "美元"))
            if "服务业领域重点建设项目" in condition:
                res["zhibiao"].append(("所属领域", "属于", "服务业", ""))
            # if "年龄" in condition:
            #     rs = re.search("税前年薪(不低于([0-9]+?)万元)", condition)
            #     if rs:
            #         res["zhibiao_person"].append(("年龄", rs.group(1)))


        return res

    def parse_precision_v2(self, conditions):
        pass

    # 解析联系方式
    def parse_contact_info(self):
        contact_infos = []
        for sub_sentence in self.half_struction["contact_infos"]:
            # print([sub_sentence])
            rs = re.search("([\u4e00-\u9fa5]\u2003[\u4e00-\u9fa5])+\s([0-9]{8})", sub_sentence)
            if rs:
                contact_infos.append((rs.group(1), rs.group(2)))
        return contact_infos

    def clear_data(self):
        self.content = []
        self.root = TwoHeadTree("root")

    # 展示文本结构
    def display_document(self):
        for span in self.content:
            print("\t"*span["title_level"] + span["title"])


def check_is_project_title(input_str):
    if "申报" in input_str:
        return False
    return True
