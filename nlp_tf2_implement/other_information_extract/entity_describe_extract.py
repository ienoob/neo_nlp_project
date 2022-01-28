#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/5/17 23:12
    @Author  : jack.li
    @Site    : 
    @File    : entity_describe_extract.py

    任务名称：实体 实体描述分析，目的是抽取出文本中的实体以及对应的实体描述。

"""
import os
import json
import jieba
import time
import re
from ltp import LTP
import numpy as np
import hanlp
from nlp_applications.data_loader import load_json_line_data
from nlp_applications.utils import load_word_vector
from nlp_applications.ner.evaluation import metrix_v2
from utils.neo_function import split_str
from entity_describe_data_p1 import d
from numpy import dot
from numpy.linalg import norm
from entity_describe_data_p2 import generater_label_single_file, generate_sohu, generator_weibo_list, generate_label_data
import multiprocessing


class EntityDescribeExtractByRoleAnalysisV1(object):
    """
        基于ltp 角色分析进行实体和实体描述信息抽取
    """

    def __init__(self, ltp_model_path="tiny"):
        self.ltp = LTP(ltp_model_path)

    def single_sentence(self, input_sentence, ind=0):
        seg, hidden = self.ltp.seg([input_sentence])
        words = seg[ind]

        pos = self.ltp.pos(hidden)[ind]
        roles = self.ltp.srl(hidden, keep_empty=False)[ind]
        filter_p = {"是", "为"}
        role_list = ["A0", "A1", "A2", "A3", "A5"]
        # print(words)
        spo_list = []
        for role in roles:
            r_indx, r_list = role

            p_value = words[r_indx]
            r_list = list(filter(lambda x: x[0] in role_list, r_list))
            if len(r_list) != 2:
                continue
            sub = r_list[0]
            obj = r_list[1]

            if sub[0] not in role_list:
                continue
            if obj[0] not in role_list:
                continue
            if sub[2] >= r_indx:
                continue
            if obj[1] <= r_indx:
                continue
            # 谓语过滤
            if p_value not in filter_p:
                continue
            if p_value == "为":
                sub, obj = obj, sub

            # 词性过滤
            if pos[sub[2]] not in ["n", "nz"]:
                continue

            sub_value = words[sub[1]:sub[2] + 1]

            obj_value = words[obj[1]:obj[2] + 1]

            # print("".join(sub_value), p_value, "".join(obj_value))
            spo_list.append(("".join(sub_value), p_value, "".join(obj_value)))

        return spo_list

    def extract_info(self, input_sentence_list):
        """ 抽取实体描述信息
        Args:
            input_sentence_list:

        Returns:
            entity_describe_res: List[{"sentence": xxx, "entity": xxx, "describe":xxx}]

        """
        entity_describe_res = []
        for i, sentence in enumerate(input_sentence_list):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            if len(sentence) > 100:
                continue
            if not re.fullmatch("^[\u4e00-\u9fa5_a-zA-Z]{1,15}是.+$", sentence):
                continue
            out_spo_list = self.single_sentence(sentence)

            for spo in out_spo_list:
                entity_describe_res.append({"sentence": sentence, "entity": spo[0], "describe": spo[2]})
        return entity_describe_res

    # def single_sentence_v2(self, input_sentence):
    #     sentence_feature = [(cut.DEPREL, cut.LEMMA) for cut in HanLP.parseDependency(input_sentence)]
    #     if sentence_feature[0][0] != "主谓关系":
    #         return True
    #     if ("核心关系", "是") not in sentence_feature:
    #         return True
    #     return False

    def multi_extract_info(self, input_sentence_list):
        pool = multiprocessing.Pool(processes=3)
        spo_res = []
        for i, sentence in enumerate(input_sentence_list):
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue

            out_spo_list = pool.apply_async(self.single_sentence, (sentence,))
            # out_spo_list = self.single_sentence(sentence)
            spo_res.append(out_spo_list)
            # spo_res.append((sentence, out_spo_list))
        pool.close()
        pool.join()

        spo_res = [spo.get() for i, spo in enumerate(spo_res)]
        return spo_res


class RuleValueTree(object):

    def __init__(self, value=None):
        self.value = value
        self.next = dict()
        self.is_leaf = 0

rule_list = [
            {
                "sub_word_num": 1,
                "sub_pos_full": "n",
                "sub_dep_full": "top",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 1,
                "sub_pos_full": "ns",
                "sub_dep_full": "nsubj",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 2,
                "sub_pos_full": "n-n",
                "sub_dep_full": "nn-top",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 1,
                "sub_pos_full": "n",
                "sub_dep_full": "nsubj",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 1,
                "sub_pos_full": "nx",
                "sub_dep_full": "top",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 1,
                "sub_pos_full": "n",
                "sub_dep_full": "dep",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 2
            },{
                "sub_word_num": 1,
                "sub_pos_full": "ns",
                "sub_dep_full": "nsubj",
                "pre_dep": "conj",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 4
            },{
                "sub_word_num": 2,
                "sub_pos_full": "t-t",
                "sub_dep_full": "nn-top",
                "pre_dep": "conj",
                "obj_pos": "q",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 2,
                "sub_pos_full": "ns-n",
                "sub_dep_full": "nn-top",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 2,
                "sub_pos_full": "ns-n",
                "sub_dep_full": "nn-nsubj",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            },{
                "sub_word_num": 1,
                "sub_pos_full": "n",
                "sub_dep_full": "dep",
                "pre_dep": "root",
                "obj_pos": "n",
                "obj_dep": "attr",
                "pre_sub_dis": 1
            },{
                "sub_word_num": 1,
                "sub_pos_full": "n",
                "sub_dep_full": "top",
                "pre_dep": "root",
                "obj_pos": "vn",
                "obj_dep": "attr",
                "pre_sub_dis": 0
            }]

#
class EntityDescribeExtractByRoleAnalysis(object):
    """
        基于hanlp 进行实体和实体描述信息抽取
    """

    def __init__(self):
        model_path = "D:\\tmp\hanlp\model\close_tok_pos_ner_srl_dep_sdp_con_electra_small_20210304_135840"
        # hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH = ""
        self.HanLP = hanlp.load(model_path)
        self.filter_p = {"是"}
        self.role_list = ["ARG0", "PRED", "ARG1"]

        self.rule_root = RuleValueTree()
        input_rule_lists = [[rule["sub_word_num"],
                            rule["sub_pos_full"],
                            rule["sub_dep_full"],
                            rule["pre_dep"],
                            rule["obj_pos"],
                            rule["obj_dep"],
                            rule["pre_sub_dis"]] for rule in rule_list]
        self.build_rule_tree(input_rule_lists)

    def add_rule(self, input_rule_list):
        cur = self.rule_root
        for rule_value in input_rule_list:
            if rule_value not in cur.next:
                cur.next[rule_value] = RuleValueTree(rule_value)
            cur = cur.next[rule_value]
        cur.is_leaf = 1

    def build_rule_tree(self, input_rule_lists):
        for input_rule_list in input_rule_lists:
            self.add_rule(input_rule_list)

    def single_sentence(self, input_sentence):
        document = self.HanLP([input_sentence])
        doc_srl = document["srl"][0]
        doc_pos = document["pos/pku"][0]
        doc_word = document["tok/fine"][0]
        doc_dep = document["dep"][0]

        spo_list = self.single_document(doc_srl, doc_pos, doc_word, doc_dep)
        return spo_list

    def rule_v1(self, spo, input_document_pos, input_document_dep, input_document_word):
        sentence_dep = input_document_dep
        sentence_pos = input_document_pos
        sub = spo["s"]
        obj = spo["o"]
        pre = spo["p"]
        if spo["s"][0] in ["下图", "图片", "照片"]:
            return True
        if sub[2]-sub[1] != 1:
            return True
        if sentence_pos[sub[2] - 1] != "n":
            return True
        if sentence_pos[obj[2] - 1] != "n":
            return True
        if sentence_dep[pre[1]][1] != "root":
            return True
        if sentence_dep[sub[2] - 1][1] != "top":
            return True
        if sentence_dep[obj[2] - 1][1] != "attr":
            return True
        if pre[1] - sub[2] != 0:
            return True
        if sub[2]-1 != 0:
            return True
        return False

    def rule_v2(self, spo, input_document_pos, input_document_dep, input_document_word):
        sentence_dep = input_document_dep
        sentence_pos = input_document_pos
        sub = spo["s"]
        obj = spo["o"]
        pre = spo["p"]

        if sub[2]-sub[1] != 1:
            return True
        if sentence_pos[sub[2] - 1] != "ns":
            return True
        if sentence_pos[obj[2] - 1] != "n":
            return True
        if sentence_dep[pre[1]][1] != "root":
            return True
        if sentence_dep[sub[2] - 1][1] != "nsubj":
            return True
        if sentence_dep[obj[2] - 1][1] != "attr":
            return True
        if pre[1] - sub[2] != 0:
            return True
        # if sub[2]-1 != 3:
        #     return True
        return False

    def rule_(self, spo, input_document_pos, input_document_word, input_document_dep):
        # 主语最后一个词必须是名词
        if input_document_pos[spo["s"][2] - 1] not in ["n"]:
            return True
        if input_document_pos[spo["o"][2] - 1] not in ["n"]:
            return True
        if input_document_word[spo["s"][1]] in ["这", "后面", "我", "那些", "这些", "這", "这个", "她", "我们", "本",
                                                "该", "那", "他们", "下列", "这位", "哪个", "下面", "前", "后", "这部",
                                                "下图", "圖", "公司", "他", "现在", "今日", "今天", "这次", "你", "右图",
                                                "本文", "左边", "图", "前述", "很多", "让", "在"]:
            return True
        if input_document_word[spo["p"][1] - 1] in ["不"]:
            return True
        if input_document_dep[spo["p"][1]][1] != "root":
            return True
        if input_document_word[spo["o"][1]] in ["我", "本", "我们"]:
            return True
        if input_document_dep[spo["o"][2] - 1][0] != spo["p"][1] + 1:
            return True
        # if input_document_dep[spo["p"][1]][1] != "root":
        #     continue
        if len(spo["o"][0]) < 8:
            return True
        if spo["p"][1] < spo["s"][2]:
            return True
        if spo["p"][2] > spo["o"][1]:
            return True

        return False

    def rule(self, spo, input_document_pos , input_document_dep, input_document_word):
        # rule_use = rule_list[0]
        sentence_dep = input_document_dep
        sentence_pos = input_document_pos
        sub = spo["s"]
        obj = spo["o"]
        pre = spo["p"]

        if spo["s"][0] in ["下图", "图片", "照片", "图", "右图"]:
            return True

        cur = self.rule_root
        sub_word_num = sub[2]-sub[1]
        if sub_word_num not in cur.next:
            return True
        cur = cur.next[sub_word_num]
        sub_pos_full = "-".join(sentence_pos[sub[1]:sub[2]])
        if sub_pos_full not in cur.next:
            return True
        cur = cur.next[sub_pos_full]
        sub_dep = [item[1] for item in sentence_dep[sub[1]:sub[2]]]
        sub_dep_full = "-".join(sub_dep)
        if sub_dep_full not in cur.next:
            return True
        cur = cur.next[sub_dep_full]
        pre_dep = sentence_dep[pre[1]][1]
        if pre_dep not in cur.next:
            return True
        cur = cur.next[pre_dep]
        obj_pos = sentence_pos[obj[2] - 1]
        if obj_pos not in cur.next:
            return True
        cur = cur.next[obj_pos]
        obj_dep = sentence_dep[obj[2] - 1][1]
        if obj_dep not in cur.next:
            return True
        cur = cur.next[obj_dep]
        pre_sub_dis = pre[1] - sub[2]
        if pre_sub_dis not in cur.next:
            return True

        return False


    def single_document(self, input_document_srl, input_document_pos, input_document_word, input_document_dep=None):
        spo_list = []

        for role_part in input_document_srl:
            spo = {}
            if len(role_part) < 3:
                continue

            for role_mention, role_key, start, end in role_part:
                if role_key not in self.role_list:
                    continue
                if role_key == "PRED" and role_mention not in self.filter_p:
                    continue
                if role_key == "PRED":
                    spo["p"] = (role_mention, start, end)
                elif role_key == "ARG0":
                    spo["s"] = (role_mention, start, end)
                elif role_key == "ARG1":
                    spo["o"] = (role_mention, start, end)

            if "p" not in spo:
                continue
            if "o" not in spo:
                continue
            if "s" not in spo:
                continue
            if spo["p"][1] < spo["s"][2]:
                continue
            if spo["p"][2] > spo["o"][1]:
                continue

            if (spo["o"][2]-spo["s"][1])*1.0/len(input_document_word) < 0.6:
                # print(spo["o"], len(input_document_word))
                continue

            if self.rule(spo, input_document_pos, input_document_dep, input_document_word):
                continue
            # if input_document_word[spo["s"][1]] in ["这", "后面", "我", "那些", "这些", "這", "这个", "她", "我们", "本",
            #                                         "该", "那", "他们", "下列", "这位", "哪个", "下面", "前", "后", "这部",
            #                                         "下图", "圖", "公司", "他", "现在", "今日", "今天", "这次", "你", "右图",
            #                                         "本文", "左边", "图", "前述", "很多", "让", "在"]:
            #     continue

            spo_list.append(spo)
            break
        return spo_list

    def extract_info(self, input_data):
        if isinstance(input_data, str):
            input_sentence_list = re.split("([。？！\s+/\n])", input_data)
        elif isinstance(input_data, list):
            input_sentence_list = input_data
        else:
            raise TypeError

        def simple_filter(i_sentence):
            if len(i_sentence) < 10:
                return False
            if len(i_sentence) > 100:
                return False
            if i_sentence[-1] in ["?", "？"]:
                return False
            if "是我" in i_sentence:
                return False
            if "是你" in i_sentence:
                return False
            if i_sentence[0] == "但":
                return False
            if not re.fullmatch("^[\u4e00-\u9fa5_a-zA-Z]{1,15}是.+$", i_sentence):
                return False
            if re.fullmatch("^.+吗$", i_sentence):
                return False
            return True

        input_sentence_list = [sentence.strip() for sentence in input_sentence_list if simple_filter(sentence.strip())]
        output_documents = self.HanLP(input_sentence_list, tasks=["srl", "tok/fine", "pos/pku", "dep"])
        entity_describe_res = []
        for i, document in enumerate(output_documents.get("srl", list())):
            sentence_words = output_documents["tok/fine"][i]
            out_spo_list = self.single_document(document,
                                                output_documents["pos/pku"][i],
                                                sentence_words,
                                                output_documents["dep"][i])
            for spo in out_spo_list:
                entity_describe_res.append(({"sentence": input_sentence_list[i],
                                            "entity": spo["s"][0],
                                            "describe": "".join(sentence_words[spo["s"][1]:spo["o"][2]])},
                                            sentence_words,
                                            output_documents["dep"][i]
                                            ))

        return entity_describe_res


def test_extract_performance():
    ede_model = EntityDescribeExtractByRoleAnalysis()

    sentence_list = ["数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科，从某种角度看属于形式科学的一种。",
                     "AnyShare由上海爱数信息技术股份有限公司自主研发的一款软硬件一体化产品，主要面向企业级用户，提供非结构化数据管理方案。",
                     "江苏恒瑞医药股份有限公司是一家从事医药创新和高品质药品研发、生产及推广的医药健康企业",
                     "操作系统（，缩写作 ）是管理计算机硬件与软件资源的系统软件，同时也是计算机系统的核心与基石。"]
    data_content = "郑佩佩再见潘垒导演：他是电影界的奇才(图)摘要：潘垒：邵氏四大文艺导演之一。他青年时移居台湾，后成为《台北诗集》主编，著有《安平港》等小说，可谓一位文学导演。他执导的电影，无论是文艺或武侠片，亦仿佛有一份诗意。1963年他加盟邵氏后，便开拍以北海岸渔村为主题的电影《情人石》(1964)。不同于多数在邵氏影城搭景的作品，潘垒实地到苏澳、台南安平等地取景，动用大批临时演员，营造写实的电影风格。其他代表作包括《兰屿之歌》、《新不了情》等。今期主持：郑佩佩郑佩佩：著名女演员，曾入选本报主办的华语电影传媒大奖评选的中国电影百年百大影星，曾被誉为“武侠影后”，其主演的武侠片《大醉侠》、《金燕子》、《卧虎藏龙》等在华语电影界影响深远。今期嘉宾：潘垒潘垒：邵氏四大文艺导演之一。他青年时移居台湾，后成为《台北诗集》主编，著有《安平港》等小说，可谓一位文学导演。他执导的电影，无论是文艺或武侠片，亦仿佛有一份诗意。1963年他加盟邵氏后，便开拍以北海岸渔村为主题的电影《情人石》(1964)。不同于多数在邵氏影城搭景的作品，潘垒实地到苏澳、台南安平等地取景，动用大批临时演员，营造写实的电影风格。其他代表作包括《兰屿之歌》、《新不了情》等。这一年里我经常在内地拍戏，很少回香港的家。一直到五月份，我在海南拍完了《中国式相亲》，接到另一出连续剧《莲花雨》，是在珠海和澳门拍的。因为这两个地方离香港都很近，只要坐一个小时的船就能到，所以一没我的戏，我就会往家里跑，也因为这样，才能有时间跟老朋友们聚聚。没想到就因为这个“忙”字，错过了和何梦华导演见最后一面……之后，我悟出了一个道理，“今后想什么就赶快做，想见谁就赶快见，免得迟了一步让自己后悔”。本来接下去我那经纪人公司帮我接了部戏是去哈尔滨的，但不知怎么地一延又延。或许因为今年的夏天太热了，是上天不想我太累了，想让我多休息休息；但是我这停不住的人，突然开始惦记起我台湾的那几位长辈了。尤其是潘垒潘导演。现在连何导演都走了，当年在邵氏导过我的几位导演，就剩下这位《情人石》和《兰屿之歌》的潘垒潘导演了。在我的那本《戏非戏》里曾提到过，我这位潘垒导演是个电影界的奇才；其实对我来说，何止这样。我常在想，当初他怎么就会选上我当女主角呢？没想到四十几年后，谜底终于揭晓了。虽然说我的第一部戏，是岳枫岳老爷导演的《宝莲灯》。但因为种种原因，《宝莲灯》一直都没有完成，所以我第一部跟观众见面的戏是由潘垒导演的《情人石》，我还因为这部片拿到了一个金武士新人奖。可见潘导演在我的演艺生涯中，应该说也是很重要的一位。我常在想，当初他怎么就会选上我当女主角呢？说实在的，我甚至都不记得去台湾拍《情人石》之前，我和他在公司有见过面呢。直到最近，我才找到个机会问他，“潘导演啊，当年公司那么多新人，你怎么就选上我当了《情人石》的女主角呢？”没想到他却说，“我还帮你付过车钱呢，我就知道我这钱白给了，你从来都没往心里去……”这倒让我有点不好意思了，“是吗？有这回事吗？对不起，我真的一点都不记得了。”“那时候你扎了个马尾，走路一蹦一跳的，一身的青春气息，不怎么爱答理人的。”到底是做导演的，观察细腻，一语就道破，我年轻时准就是这个样，“那年头，从清水湾邵氏片厂到九龙市区，可不方便了，如果没搭上公司车，又没赶上巴士，那就只能坐小巴了。那天你大概是要去九龙，我也正好要去九龙，一前一后地，我就跟在你的后面走；你上了小巴，我也上了那辆小巴。虽然没人给我们介绍过，总还是同事一场，所以我顺手帮你把车钱给付了，你只是跟我笑笑，大概就算表示谢谢了，却一路上也没跟我言语过，也没当一回事，好像我是应该帮你付车钱的……”“是吗？有这样的事吗？”他这么说我可就更不好意思了。没想到的是，就我这副样子，还给他留下了一个很深刻的印象，“正巧公司让我把我的那本小说《安平港》改编成电影，不过指定女主角要我起用新人。当时邵氏的新人一大把，但是我却一个也不认识；老板逼着我作决定，问我想起用哪个新人，我突然就想起了你……”哈哈，没想到四十几年后，谜底终于揭晓了。潘导演那对大眼睛，可是充满了智能，所以他在文玲姐身上看到了那种可以藏在家里的贤德。“潘导演，你还记不记得，拍《情人石》的时候，我一下飞机就被你抓去当黄大哥(黄宗迅)和焦姣姐的伴娘了。”“哈哈哈，那是因为焦姣要跟你黄大哥一起去台南出外景，那时比较保守，他们还不是夫妇，没名没分的一起去似乎不太好。我就建议他们干脆马上结婚，就你来的那天晚上，就那餐饭桌上……”“那看起来我和他们缘分还真不浅呢！一直到现在都还忍不住要提这事。后来焦姣都警告我了，我还乱嚷嚷，曾江都为这生气了，让我别哪壶不开偏提哪壶，哈哈哈……”但是我们每一次见面，都还是得提这个老掉牙的往事。我们这些上了年纪的人聚在一起，就爱提往事，这一提就没完没了了。“潘导演啊，你给我说说，你和文玲姐又是怎么认识的呢？”我口中的文玲姐是潘导演的太太，她永远是让我感到最自在的人。当年在《情人石》里她演另一个女主角，我的情敌。可不管在戏里我和文玲姐是什么关系，但在真实生活中我喜欢她———我第一次见到她，就打心里喜欢她了。她那时不但已经是潘太太，而且还是两个孩子的母亲，然而从她那双大眼睛里，我看到的是那种出污泥而不染的“真”。而这种“真”，现在差不多已经绝版，就算是当年，也已经很少见了。不用说，当年的潘导演，也是被这双大眼睛迷上了。“我认识文玲时，她很年轻，是来考‘台湾中影公司’的新人，我当时是其中一个考官。我悄悄地告诉她，她的最大优点是那双大眼睛，所以得多用用眼睛来吸引其他考官。”“她凭那双大眼睛考上了？”“没有。她傻乎乎的，哪懂得怎么用她那双大眼睛。”接着他就学文玲眨巴眼睛的样子，我们就一起大笑起来。“那文玲姐，她后来是怎么开始拍戏的呢？”“哦，后来有人找我拍台语片，我就让她当我的女主角，她就成了台语片的明星了。其实她也不怎么在乎拍不拍戏的，基本上她就只拍过我的戏。”讲起这个，不管是什么时候，不论是过去还是现在，潘导演的脸上都充满了男人的那种得意。其实潘导演也有一双大得惊人的眼睛，尤其是当我们在兰屿拍《兰屿之歌》时，他整个脸都晒成黑炭，就剩下那双大眼睛了；再配上他那副又干又瘦的身子骨，活像个瘦猴子。不过他那对大眼睛，可是充满了智能，所以他在文玲姐身上一眼看到的，不只是那种“真”，还有那种可以藏在家里的贤德。我从来都只是以为，潘导演娶到了一位贤淑的好妻子；没想到原来文玲姐嫁的，是这么一位好丈夫。我上次见他们夫妇时，应该是一年前的事了；我得知他们搬到台湾去住了，又听说文玲姐中风了，所以找了个机会去台湾探望他们。记得那次潘导演约了我在地铁出口碰头，一见面没带我去看文玲姐，先带我去了一间咖啡馆。他仔仔细细地把文玲姐的情况跟我介绍了一遍，他是担心我脸上的异样，会让文玲姐不安。“她已经是第二次开刀了，可以说是一种奇迹，她的脑子非常清醒。糟糕的是，有一天她不小心掉下了床，把脊椎骨摔折了，现在连坐都成问题，只能整天躺在安养院的床上。”其实文玲姐就像潘导演说的那样，虽然身子不能动，脑子还是清楚得很，她不但记得我是谁，还把她身上那块白玉给了我做念想。不论躺着还是坐着，文玲姐的手都没离开过潘导演，抓紧了不放，口中还一直叫着“爸爸，爸爸”。让我感动的是潘导演，他跟我说，“我现在每天的生活就是两个宝贝，一个是这个大宝贝(他指了一下身边的文玲姐)，回到家里我就玩我的小宝贝。”他的小宝贝就是“玉”，那不只是嗜好那么简单，他不但收藏，而且还出书研究，应该可以说是一位专家了。大宝贝几乎已经占了他大部分的时间，“我不去，她就不吃饭。我担心安养院的饭不够营养，每天两餐都是我亲手给她做，然后再一口一口喂她吃。”我从来都只是以为，潘导演娶到了一位贤淑的好妻子；没想到的是，原来文玲姐嫁的，会是这么一位好丈夫。记得当年，很多人在一起的时候，文玲姐不怎么说话，因为话都让潘导演一个人说完了。所以总给人一种错觉，潘导演就像很多台湾男子那样，很大男人主义；再加上电影界桃色新闻多，这段婚姻很多人就没怎么看好。这么多年过去了，也没听说过潘导演有什么艳遇；不过，能把这四五十年的婚姻维持下来，显然这个当妻子的，需要更多忍耐。不过潘导演对我说，“你信不信，我们几十年夫妻，从来都没吵过架。我脾气不好，怎么发脾气，她都不搭腔。你知道，她本来就不怎么讲话，反而现在比较话多一些。”所谓的话多一些，也就是不断重复的那两句话，“爸爸，我好想你啊。”然后就是一次次对我说，“他对我很好，他是一个好人。”我和潘导演这次也是约在地铁口，约好了上午十点。也不知怎么的，正好是十点整，一分也不早，一分也不迟，我就出现在我们约定的地铁口了。那天我从地下往上走的时候，抬头看见潘导演站在路口，他指指戴在手上的手表，竖起大拇指；等到我走到他身边，他对我说，“好样的，几十年不变，永远是最守时的一个。”我在想，或许，能让我几十年都不变的，是这些长辈的这句话，时时刻刻都在提醒我，永远要守时。这回，他迫不期待地把我领到了安养院。我们开门进去，客厅里挤满了一张张轮椅，没等我仔细去找，潘导演把我领进里屋，拉了张椅子，招呼我坐在床边，“你坐一下，我去把她带进来。”有些话出自这对老夫妇的口中，你不会觉得一丝一毫的肉麻，只会被他们感动。我打量了一下，这张文玲姐又躺了一年的床，似乎没什么改变。事实上，文玲姐却是大有进步，脸没上一次那么肿了，似乎人也比上一次清醒一些；最大的进步是，这一回她可以坐起来了。潘导演兴奋地告诉我，“这可是奇迹。现在她还不能走，医生说只要她能走，她就可以回家了。要知道进来这儿的人，几乎是没人能出去的。”文玲姐的话，也比一年前多了两句。她说“爸爸是个好人”，还告诉我“爸爸对我很好”，只是她重复了又重复，潘导演给她数了一下，都超过了十次。可能是因为我在，潘导演有点不好意思，“好话说一次就够了，说多了别人以为你是在拍马屁了。”文玲姐居然会反驳，“我说的是真的，那你说你对我好不好？”“麻麻地了。”接下去再说到这个问题时，她改口了“爸爸对我麻麻地。”“爸爸对你这么好，怎么是麻麻地呢？”“他自己说的，他对我麻麻地。”难怪潘导演又一次提到，“你别看她行动不方便，头脑可清楚得很呢。”可不是嘛。她听见临床的那位老人，不断地发出呻吟声，于是皱起了眉头，用广东话说道，“我点算？爸爸，我点算？”潘导演用手试着去抚平文玲姐的眉头，“别皱眉头，皱了就不好看了。”“爸爸，我好想你啊……”“我也好想你啊。”这话出自这对老夫妇的口中，你不会觉得一丝一毫的肉麻，只会被他们感动。文玲姐突然又来了一句，“我好苦啊……”“文玲姐，你怎么会苦呢？你有一个那么好的老公，谁都没你有福气呢。”没一会她又来了，“爸爸，你说我点算啊？”我在一旁多了句嘴，“你得听爸爸的话，好好练走路。等你能走了，就可以跟爸爸回家了。”她听到“回家”两个字，眼睛都发亮了“爸爸，回家，我要回家……”“对啊，你今天练了没有啊？”潘导演回过头对我说，“她现在可不听话呢，尤其是吃饭，有我在她就耍赖。”“那你现在不喂她吃饭了？”“没喂半年了，医院也不让我煮了。他们让我看他们给病人的伙食，都特别注意营养，绝对不比我弄的差。”我们这边谈着，文玲姐那边和照顾他们的菲律宾护士用英语聊起天来。“你看看，你文玲姐还会说英文呢。”“是啊，不只是‘yes’、‘no’的，还成句成句的。她什么时候学的？”“我也不知道什么时候，她瞒着我自己去‘格致’报名的。”潘导演越挖掘就越发觉这块“宝”的分量，就越宝贝这个“宝贝”了。他把那杯子打开，“我真的还没喝汤呢。你怎么也没提醒我呢？”不知怎么的，我鼻子有点酸。医院开饭时间早，才十一点护士已经来叫要开饭了。潘导演对文玲姐说，“我带佩佩去吃饭了，你跟佩佩说再见吧。”文玲姐依依不舍，还死拉着潘导演的手不放“爸爸，你吃了饭要赶快回来啊。”“那你要答应我乖乖吃饭哦。”就这样连哄带骗的，潘导演带我离开了安养院，“我每天中午都在一间日本餐厅吃个沙律，喝碗汤。”“我这吃素的，吃沙律、喝汤正合我意，只要他们有素汤。”说着说着我们就走进了一家快餐店，我看了一下他们挂着的餐牌，才发现那还真是家日本式的快餐店。轮到潘导演点菜时，他首先就问那伙计，“你们的汤有素的吗？”那伙计指了指餐牌，“这蘑菇汤就是素的。”“好极了，”潘导演转过头来问我，“怎么样，蘑菇汤可以吗？你还想要什么？”结果我们每人点了一个沙律，一杯蘑菇汤。他怕我不够，还一人点了一个果冻，并且还告诉伙计我们俩就在小店吃。东西来齐了，他拿着食物就往楼上走。不知怎么的，我觉得这个地方似曾相识；到了楼上我才想起，这就是上回他带我来喝咖啡的地方。我们一边吃，一边还是在聊着文玲姐的事，“文玲姐能走了，你真把她带回家吗？”“她想回香港的家。”“哦，在西贡的那个家吗？”“不行。如果要回香港的话，也不能住那儿，一定得搬。”“那倒是。那在楼上，就算文玲姐好了，能走了，这楼上楼下，还是不方便的。”“我们小儿子想让我们搬去北京，他住在北京。几个孩子里数他的环境最好，文玲姐进医院开刀，所有的手术费都是我们这个小儿子付的。”做父母的最大的幸福，就是能以儿女自豪。“那也不错啊。”我附和着他的话。“他在北京就住在飞机场附近的小区。”“哦，我去过那儿。那儿的环境不错，空气也比市区里好。”我为他们高兴，好像问题一下子都可以解决了，“而且在内地，要请个工人也容易一些。就算是文玲姐好了，怎么说你一个人照顾是不行的，起码得请个二十四小时的看护。”“可是……”“还可是什么呀！”我都开始急了。“我担心北京的气候，我不习惯，文玲也无法接受。”“没事的。北京冬天冷，可是冬天屋子里都有暖气的。”“但是北京的夏天气温又特别高……”我向来吃东西是狼吞虎咽的，尤其是汤，我一口气趁热就喝了。那沙律用的是日本式的沙律酱，特好吃，三口两口就吃完了。然后把果冻当着甜点，吃得还满舒服的。潘导演吃得比我斯文，那汤可能有点烫，他就先吃沙律；然后他把果冻也吃了，我心想，大概是觉得那汤还是太烫了；怎么知道果冻吃完，他就说可以走了。“那汤呢？你吃不下了？准备留着当下午茶了？”我半开玩笑地对他说。他这才把那杯子打开了，“我真的还没喝汤呢。你怎么也没提醒我呢？”我看着他把那杯汤喝了下去，不知怎么的，鼻子有点酸……潘导演比一年前老多了。"
    sentence_list += data_content.split("。")

    data_size = 0
    byte_size = 0
    sentence_num = 0
    start_a_time = time.time()
    out_info = ede_model.extract_info(sentence_list)
    # print(out_info)


    data_size += sum([len(sentence) for sentence in sentence_list])
    byte_size += sum([len(sentence.encode()) for sentence in sentence_list])
    # print(sentence_list[0].encode())
    sentence_num += len(sentence_list)

    cost_time = time.time() - start_a_time

    cost_time += 0.000000000000001

    print("data size {}".format(data_size))
    print("sentence num {}".format(sentence_num))
    print("cost time {} s".format(cost_time))
    print("{} sentence/s".format(sentence_num / cost_time))
    print("{} char/s".format(data_size / cost_time))
    print("{} byte/s".format(byte_size / cost_time))


def eval_metrix(hit_num, true_num, predict_num):
    recall = (hit_num + 1e-8) / (true_num + 1e-3)
    precision = (hit_num + 1e-8) / (predict_num + 1e-3)
    f1_value = 2 * recall * precision / (recall + precision)

    return {
        "recall": recall,
        "precision": precision,
        "f1_value": f1_value
    }

entity_describe = [
    {"entity": "江苏恒瑞医药股份有限公司", "sentence": "江苏恒瑞医药股份有限公司是一家从事医药创新和高品质药品研发、生产及推广的医药健康企业"}
]

weibo_path = "D:\data\语料库\weibo_2019-05-18_10.30.41.txt\weibo_2019-05-18_10.30.41"
wiki_path = "D:\data\语料库\wiki_zh_2019\wiki_zh\AA"
sohu_path = "D:\data\语料库\sohu-20091019-20130819\sohu-20091019-20130819"



def test_hanlp():

    ede_model = EntityDescribeExtractByRoleAnalysis()

    data_size = 0.0
    byte_size = 0.0
    cost_time = 0.0

    data_iter = generator_weibo_list(weibo_path)
    sentence = 0
    hit_num = 0.0
    predict_num = 0.0
    real_num = 0.0
    file_name = "weibo_rule4.jsonline"
    with open(file_name, "w") as f:
        f.write("")
    for data_content in data_iter:
        real_num += 1

        data_size += len(data_content)
        byte_size += len(data_content.encode())
        start = time.time()
        data_list = list(split_str(data_content, {"。", "？", "！", "!", "?", "\r", "\n", "//"}))
        # print(len(data_list))
        e_res = ede_model.extract_info(data_list)
        if e_res:

            print(e_res[0][0])
            print(e_res[0][1])
            print(e_res[0][2])
        for es in e_res:
            predict_num += 1
            with open(file_name, "a+", encoding="utf-8") as f:
                # print(json.dumps(es[0]))
                f.write(json.dumps(es[0],  ensure_ascii=False)+"\r")
        sentence += len(data_list)
        if sentence > 100000:
            break
        cost_time += time.time() - start + 0.000000000000001

    print("hit num {0} predict num {1} real num {2}".format(hit_num, predict_num, real_num))

    print("data size {}".format(data_size))
    print("sentence num {}".format(sentence))
    print("cost time {} s".format(cost_time))
    print("{} sentence/s".format(sentence / cost_time))
    print("{} char/s".format(data_size / cost_time))
    print("{} byte/s".format(byte_size / cost_time))


def test_hanlp_wiki():

    ede_model = EntityDescribeExtractByRoleAnalysis()

    data_size = 0.0
    byte_size = 0.0
    cost_time = 0.0

    data_iter = generate_label_data()
    sentence = 0
    hit_num = 0.0
    predict_num = 0.0
    real_num = 0.0
    file_name = "weiki.jsonline"
    with open(file_name, "w") as f:
        f.write("")
    for data_dict in data_iter:
        data_entity = data_dict["entity"]
        data_content = data_dict["sentence"]
        real_num += 1
        data_size += len(data_content)
        byte_size += len(data_content.encode())
        start = time.time()
        data_list = list(split_str(data_content, {".", "。", "？", "！", "!", "?", "\r", "\n", "//"}))
        # print(len(data_list))
        e_res = ede_model.extract_info(data_list)

        for es in e_res:
            predict_num += 1
            if es[0]["entity"] == data_entity:
                hit_num += 1

            with open(file_name, "a+", encoding="utf-8") as f:
                # print(json.dumps(es[0]))
                f.write(json.dumps(es[0],  ensure_ascii=False)+"\r")
        sentence += len(data_list)
        cost_time += time.time() - start + 0.000000000000001

    print("hit num {0} predict num {1} real num {2}".format(hit_num, predict_num, real_num))
    print(eval_metrix(hit_num, real_num, predict_num))

    print("data size {}".format(data_size))
    print("sentence num {}".format(sentence))
    print("cost time {} s".format(cost_time))
    print("{} sentence/s".format(sentence / cost_time))
    print("{} char/s".format(data_size / cost_time))
    print("{} byte/s".format(byte_size / cost_time))


# if __name__ == "__main__":
#     ede_model = EntityDescribeExtractByRoleAnalysis()
#
#     res = ede_model.single_sentence("计算机科学（，有时缩写为）是系统性研究信息与计算的理论基础以及它们在计算机系统中如何与应用的实用技术的学科")
#     print(res)

if __name__ == "__main__":
    # for item in generater_label_single_file(29):
    #     print(item)

    # edebra = EntityDescribeExtractByRoleAnalysis()
    # generator = generator_weibo_list(weibo_path)
    # i = 0
    # max_num = 100
    # for weibo in generator:
    #     sentence_list = list(split_str(weibo, {"。", "？", "！", "!", "?", "\r", "\n"}))
    #     # print(sentence_list)
    #     extract_info = edebra.extract_info(sentence_list)
    #     if extract_info:
    #         print(extract_info)
    #     else:
    #         with open("D:\\tmp\entity_describe\\negative_{}.txt".format(i), "w", encoding="utf-8") as f:
    #             f.write("\n".join(sentence_list))
    #
    #     i += 1
    #     if i > max_num:
    #         break
    test_hanlp()
    #
    # for sentence in generate_label_data():
    #     print(sentence)
    #     break




    """
        按照？。！\n \s 等进行分句，要保留句子最后的标点符号

        句子过短，少于10个字符 => 过滤
        句子很长，长于100个字符 => 过滤
        谓语不是 【是】 => 过滤
        主语最后一个词必须是名词
        谓語最后一个词必须是名词

        什么是白血病【？】 => 问号结尾，过滤掉
        钱爸是【我】觉得性格特别好的一个人 => 具有主观色彩，过滤掉
        河豚拉面的高汤是【我】喝过的世界上最好喝的汤

        人是一堆无用的热情，盛放爱意的容器 => 散文中存在的，如何处理暂定
        
        {上海市}轻工业是上海发展最早也是最为成熟的工业部门 => 缺少定语
        
        人工智能目前仍然是{该}领域的长远目标 => 找到指代 

        202106151517 測試結果：
            hit num 49.0 predict num 51.0 real num 881.0
            data size 47122.0
            sentence num 1149
            cost time 19.53078532219031 s
            58.830199658922034 sentence/s
            2412.703801851805 char/s
            6669.214670646554 byte/s

    """

    negative_sentence = [
        "城市是阿尔巴尼亚地方治理的第一级行政区，以便于地方政府管埋和负责当地的事务。",
        "原意是模仿敌人被杀后，头颅被挂在长竿上的样子。",
        "很多本主是神话、传说、历史中的著名人物。",
        "一派是以闵妃为首的外戚集团，另一派则是要求改革的士大夫激进派。",
        "温家宝不懂经济是被批评的另一个重要因素，温家宝曾经出卖赵紫阳，因此被人评价是一个两面三刀的人。",
        ""
    ]


