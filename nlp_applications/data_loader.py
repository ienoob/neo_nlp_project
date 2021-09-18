#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/29 23:13
    @Author  : jack.li
    @Site    : 
    @File    : data_loader.py

"""
import json
import jieba
import numpy as np



class LoaderDataSet(object):
    pass


class LoadMsraDataV1(object):

    def __init__(self, path):
        self.train_sentence_list = self.get_data(path+"/train/sentences.txt")
        self.train_tag_list = self.get_data(path+"/train/tags.txt")
        self.test_sentence_list = self.get_data(path+"/test/sentences.txt")
        self.test_tag_list = self.get_data(path+"/test/tags.txt")

        self.word2id = dict()
        self.tag2id = dict()

        for i, sentence in enumerate(self.train_sentence_list):
            assert len(sentence) == len(self.train_tag_list[i])

        for i, sentence in enumerate(self.test_sentence_list):
            assert len(sentence) == len(self.test_tag_list[i])

        self.labels = ["O", "B-ORG", "I-PER", "B-PER", "I-LOC", "I-ORG", "B-LOC"]

    def get_data(self, data_path):
        data_list = []
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.read()

        for dt in data.split("\n"):
            dt = dt.strip()
            if len(dt) == 0:
                continue
            d_list = []
            for w in dt.split(" "):
                d_list.append(w)
            data_list.append(d_list)
        return data_list


class LoadMsraDataV2(object):

    def __init__(self, path):
        self.train_sentence_list, self.train_tag_list = self.get_data(path + "word_level.train.jsonlines")
        self.test_sentence_list, self.test_tag_list = self.get_data(path+"word_level.test.jsonlines")

        self.labels = ["O", "B-AGE", "I-AGE", "B-ANGLE", "I-ANGLE", "B-AREA", "I-AREA", "B-CAPACTITY", "I-CAPACTITY",
                       "B-DATE", "I-DATE", "B-DECIMAL", "I-DECIMAL", "B-DURATION", "I-DURATION",
                       "B-FRACTION", "I-FRACTION", "B-FREQUENCY", "I-FREQUENCY",  "B-INTEGER", "I-INTEGER",
                       "B-LENGTH", "I-LENGTH", "B-LOCATION", "I-LOCATION", "B-MEASURE", "I-MEASURE",
                       "B-MONEY", "I-MONEY", "B-ORDINAL", "I-ORDINAL", "B-ORGANIZATION", "I-ORGANIZATION",
                       "B-PERCENT", "I-PERCENT",  "B-PERSON", "I-PERSON", "B-PHONE", "I-PHONE",
                       "B-POSTALCODE", "I-POSTALCODE", "B-RATE", "I-RATE", "B-SPEED", "I-SPEED",
                       "B-TEMPERATURE", "I-TEMPERATURE", "B-TIME", "I-TIME", "B-WEIGHT", "I-WEIGHT",
                       "B-WWW", "I-WWW"]
        # self.label2id = {"pad": 0}
        self.label2id = {}
        for la in self.labels:
            if la not in self.label2id:
                self.label2id[la] = len(self.label2id)
        self.id2label = {v:k for k, v in self.label2id.items()}

    def get_data(self, path):
        data_list = []
        tag_list = []

        with open(path, "r", encoding="utf-8") as f:
            data = f.read()

        for dt in data.split("\n"):
            if len(dt.strip()) == 0:
                continue

            tag = []
            sentence_content = []
            dt_json = json.loads(dt)

            sentence = dt_json["sentences"][0]
            ner = dt_json["ner"]

            # sentence_content = "".join(sentence[0][0])

            ner = ner[0]
            for i, d in enumerate(sentence):
                if len(ner) == 0:
                    sentence_content += list(d)
                    tag += ["O"] * len(d)
                elif i < ner[0][0]:
                    sentence_content += list(d)
                    tag += ["O"]*len(d)
                elif i == ner[0][0]:
                    sentence_content += list(d)
                    tag += ["B-{}".format(ner[0][2])] + ["I-{}".format(ner[0][2])]*(len(d)-1)
                elif  i <= ner[0][1]:
                    sentence_content += list(d)
                    tag += ["I-{}".format(ner[0][2])] * len(d)

                if len(ner) and i == ner[0][1]:
                    ner.pop(0)
            assert len(sentence_content) == len(tag)
            data_list.append(sentence_content)
            tag_list.append(tag)

        return data_list, tag_list


# 关系分类
class LoaderSemEval2010Task8(object):

    def __init__(self, data_path):

        self.data_path = data_path
        self.label = ["Other", "Cause-Effect", "Component-Whole",  "Entity-Destination", "Product-Producer",
                      "Entity-Origin", "Member-Collection", "Message-Topic",
                      "Content-Container", "Instrument-Agency"]


class LoaderBaiduKg2019RealtionExtraction(object):

    def __init__(self, data_path):

        self.train_path = data_path+"\\train_data.json"
        self.dev_path = data_path+"\\dev_data.json"

        self.data_schema = data_path+"\\all_50_schemas.json"


    def get_train_data(self):
        with open(self.data_schema, "r", encoding="utf-8") as f:
            schema = f.read()

        with open(self.train_path, "r", encoding="utf-8") as f:
            train_data = f.read()

        with open(self.dev_path, "r", encoding="utf-8") as f:
            dev_data = f.read()

        schema_data = schema.split("\n")
        self.relation_dict = dict()
        for schema in schema_data:
            if not schema.strip():
                continue
            schema = json.loads(schema)
            predicate = schema["predicate"]
            if predicate not in self.relation_dict:
                self.relation_dict[predicate] = len(self.relation_dict)


        train_data_list = train_data.split("\n")
        dev_data_list = dev_data.split("\n")

        self.word_index = {
            "pad": 0,
            "unk": 1
        }
        left_word = []
        right_word = []
        mid_word = []
        left_pos_1 = []
        left_pos_2 = []
        right_pos_1 = []
        right_pos_2 = []
        mid_pos_1 = []
        mid_pos_2 = []
        label = []
        for data in train_data_list:
            if not data.strip():
                continue
            data = json.loads(data)
            text = data["text"]
            left_word_sub = []
            right_word_sub = []
            mid_word_sub = []
            # left_pos_1_sub = []
            # left_pos_2_sub = []
            # right_pos_1_sub = []
            # right_pos_2_sub = []
            # mid_pos_1_sub = []
            # mid_pos_2_sub = []

            for t in text:
                if t not in self.word_index:
                    self.word_index[t] = len(self.word_index)
            for spo in data["spo_list"]:
                object_value = spo["object"]
                subject_value = spo["subject"]
                predicate = spo["predicate"]
                try:
                    ob_ind = text.index(object_value)
                    sub_ind = text.index(subject_value)
                except Exception as e:
                    continue

                left_ind = -1
                left_end = -1
                right_ind = -1
                right_end = -1
                if sub_ind > ob_ind:
                    left_ind = ob_ind
                    left_end = ob_ind+len(object_value)
                    right_ind = sub_ind
                    right_end = sub_ind+len(subject_value)
                elif sub_ind < ob_ind:
                    left_ind = sub_ind
                    left_end = sub_ind+len(subject_value)
                    right_ind = ob_ind
                    right_end = ob_ind+len(object_value)
                else:
                    print(object_value, subject_value, sub_ind, ob_ind)


                for t in text[:right_end]:
                    left_word_sub.append(self.word_index[t])
                left_pos_1_sub = list(range(len(text[:right_end])))
                # left_pos_1_sub = [ii-left_ind for ii in left_pos_1_sub]
                left_pos_2_sub = list(range(len(text[:right_end])))
                # left_pos_2_sub = [ii-right_ind for ii in left_pos_2_sub]

                left_word.append(left_word_sub)
                left_pos_1.append(left_pos_1_sub)
                left_pos_2.append(left_pos_2_sub)

                for t in text[left_ind:right_end]:
                    mid_word_sub.append(self.word_index[t])
                mid_pos_1_sub = list(range(len(text[left_ind:right_end])))
                mid_pos_2_sub = list(range(len(text[left_ind:right_end])))
                # mid_pos_2_sub = [ii+left_ind-right_ind for ii in mid_pos_2_sub]

                mid_word.append(mid_word_sub)
                mid_pos_1.append(mid_pos_1_sub)
                mid_pos_2.append(mid_pos_2_sub)

                for t in text[left_ind:]:
                    right_word_sub.append(self.word_index[t])
                right_pos_1_sub = list(range(len(text[left_ind:])))
                right_pos_2_sub = list(range(len(text[left_ind:])))
                # right_pos_2_sub = [ii+left_ind-right_ind for ii in right_pos_2_sub]

                right_word.append(right_word_sub)
                right_pos_1.append(right_pos_1_sub)
                right_pos_2.append(right_pos_2_sub)

                label.append([self.relation_dict[predicate]])

        return left_word, right_word, mid_word, left_pos_1, left_pos_2, right_pos_1, right_pos_2, mid_pos_1, mid_pos_2, label


class Document(object):

    def __init__(self, input_id, input_text, input_text_id, input_entity_list, input_relation_list):
        self._id = input_id
        self._raw_text = input_text
        self._text_id = input_text_id
        self._entity_list = input_entity_list
        self._relation_list = input_relation_list

    @property
    def id(self):
        return self._id

    @property
    def raw_text(self):
        return self._raw_text

    @property
    def text_id(self):
        return self._text_id

    @property
    def entity_list(self):
        return self._entity_list

    @property
    def relation_list(self):
        return self._relation_list


class Entity(object):

    def __init__(self, input_id, input_text, input_start, input_end):
        self._id = input_id
        self._entity_text = input_text
        self.size = input_end-input_start
        self._start = input_start
        self._end = input_end

    @property
    def id(self):
        return self._id

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def entity_text(self):
        return self._entity_text


class Relation(object):

    def __init__(self, input_id, input_sub: Entity, input_obj: Entity):
        self._id = input_id
        self._relation_sub = input_sub
        self._relation_obj = input_obj

    @property
    def id(self):
        return self._id

    @property
    def sub(self):
        return self._relation_sub

    @property
    def obj(self):
        return self._relation_obj


class LoaderBaiduKg2019RealtionExtractionV2(object):

    def __init__(self, data_path):
        self.train_path = data_path + "\\train_data.json"
        self.dev_path = data_path + "\\dev_data.json"

        self.data_schema = data_path + "\\all_50_schemas.json"

        with open(self.data_schema, "r", encoding="utf-8") as f:
            schema_data = f.read()
        schema_data_list = schema_data.split("\n")
        schema_data_list = [json.loads(schema) for schema in schema_data_list if schema.strip()]

        self.relation_type = len(schema_data_list)+1
        self.relation2id = {
            "no_relation": 0
        }
        self.entity2id = {
            "no_entity": 0
        }

        for relation in schema_data_list:
            subject = relation["subject_type"]
            object = relation["object_type"]
            predicate = relation["predicate"]

            if subject not in self.entity2id:
                self.entity2id[subject] = len(self.entity2id)
            if object not in self.entity2id:
                self.entity2id[object] = len(self.entity2id)
            if predicate not in self.relation2id:
                self.relation2id[predicate] = len(self.relation2id)

        with open(self.train_path, "r", encoding="utf-8") as f:
            train_data = f.read()

        train_data_list = train_data.split("\n")
        train_data_list = [json.loads(data) for data in train_data_list if data.strip()]

        self.char2id = {
            "<pad>": 0,
            "<unk>": 1
        }

        self.documents = []
        for i, train_data in enumerate(train_data_list):

            train_text = train_data["text"]
            train_text_id = []
            for tt in train_text:
                if tt not in self.char2id:
                    self.char2id[tt] = len(self.char2id)
                train_text_id.append(self.char2id[tt])

            spo_list = train_data["spo_list"]

            entity_list = []
            relation_list = []

            state = 0
            for spo in spo_list:
                sub_subject = spo["subject"]
                sub_subject_type = spo["subject_type"]
                try:
                    sub_indx = train_text.index(sub_subject)
                except Exception as e:
                    # print(e, train_text, sub_subject)
                    state = 1
                    break

                entity_sub = Entity(self.entity2id[sub_subject_type], sub_subject, sub_indx, sub_indx+len(sub_subject))

                entity_list.append(entity_sub)

                sub_object = spo["object"]
                sub_object_type = spo["object_type"]
                try:
                    obj_indx = train_text.index(sub_object)
                except Exception as e:
                    # print(e, train_text, sub_object)
                    state = 1
                    break
                entity_obj = Entity(self.entity2id[sub_object_type], sub_object, obj_indx,
                                    obj_indx + len(sub_object))
                entity_list.append(entity_obj)

                predicate_type = spo["predicate"]
                sub_relation = Relation(self.relation2id[predicate_type], entity_sub, entity_obj)
                relation_list.append(sub_relation)
            if state:
                continue
            doc = Document(i, train_text, train_text_id, entity_list, relation_list)
            self.documents.append(doc)


def load_json_line_data(input_data_path):
    with open(input_data_path, encoding="utf-8") as f:
        data = f.read()

    for da in data.split("\n"):
        if not da.strip():
            continue
        yield json.loads(da)


class LoaderDuie2Dataset(object):
    """
        DuIE2.0是业界规模最大的基于schema的中文关系抽取数据集，
        包含超过43万三元组数据、21万中文句子及48个预定义的关系类型。
        表1 展示了其中43个简单O值的关系类型及对应的例子，
        表2 展示了其中5个复杂O值的关系类型及对应的例子。
        数据集中的句子来自百度百科、百度贴吧和百度信息流文本。
    """
    def __init__(self, data_path, use_word_feature=True):
        self.schema_path = data_path + "//duie_schema//duie_schema.json"
        self.train_path = data_path + "//duie_train.json//duie_train.json"
        self.dev_path = data_path + "//duie_dev.json//duie_dev.json"
        self.test_path = data_path + "//duie_test1.json//duie_test1.json"

        self.use_word_feature = use_word_feature

        schema_data_list = load_json_line_data(self.schema_path)
        train_data_list = load_json_line_data(self.train_path)
        dev_data_list = load_json_line_data(self.dev_path)
        test_data_list = load_json_line_data(self.test_path)

        self.data_len = 0
        self.entity_max_len = 0
        self.relation2id = {
            "pad": 0
        }
        self.subject2id = dict()
        self.object2id = dict()
        self.entity2id = {
            "unk": 0
        }
        self.triple_set = set()
        self.entity_couple_set = set()
        self.max_seq_len = 0

        for schema in schema_data_list:
            predicate = schema["predicate"]
            if predicate not in self.relation2id:
                self.relation2id[predicate] = len(self.relation2id)
            subject = schema["subject_type"]
            object = schema["object_type"]["@value"]

            if subject not in self.subject2id:
                self.subject2id[subject] = len(self.subject2id)

            if object not in self.object2id:
                self.object2id[object] = len(self.object2id)

            if subject not in self.entity2id:
                self.entity2id[subject] = len(self.entity2id)

            if object not in self.entity2id:
                self.entity2id[object] = len(self.entity2id)

            self.triple_set.add((self.entity2id[subject], self.relation2id[predicate], self.entity2id[object]))
            self.entity_couple_set.add((self.entity2id[subject], self.entity2id[object]))

        self.id2entity = {v:k for k, v in self.entity2id.items()}
        self.id2relation = {v:k for k, v in self.relation2id.items()}

        self.char2id = {
            "<pad>": 0,
            "<unk>": 1
        }
        self.word2id = {
            "<pad>": 0,
            "<unk>": 1
        }

        self.documents = []
        for i, train_data in enumerate(train_data_list):
            # if i < 167933:
            #     continue
            if i == 107556:
                continue

            self.data_len += 1
            self.max_seq_len = max(self.max_seq_len, len(train_data["text"]))
            train_text = train_data["text"]
            if i == 5147:
                train_text = "学校介绍都灵大学（Università degli Studi di Torino，UNITO），位于欧洲汽车工业制造基地、意大利第三大城市、皮耶蒙特首府都灵市中心，始建于1404年，至今已有600多年的历史 作为意大利规模最大的之一 都灵大学以经济学、化学、物理学、法学、医学和心理学等基础学科的研究见长，其经济学与工商管理专业在欧洲大陆享有盛誉"

            train_text_id = []
            for tt in train_text:
                if tt not in self.char2id:
                    self.char2id[tt] = len(self.char2id)
                train_text_id.append(self.char2id[tt])
            if self.use_word_feature:
                for tword in jieba.cut(train_text):
                    if tword not in self.word2id:
                        self.word2id[tword] = len(self.word2id)

            spo_list = train_data["spo_list"]

            entity_list = []
            relation_list = []

            state = 0
            for spo in spo_list:
                sub_subject = spo["subject"]
                sub_subject_type = spo["subject_type"]
                if i == 170637:
                    if sub_subject == "":
                        continue
                if i == 70682:
                    if sub_subject == "司马迁之人格与风格 道教徒的诗人李白及其痛苦":
                        sub_subject = "司马迁之人格与风格　道教徒的诗人李白及其痛苦"
                if i == 83169:
                    if sub_subject == "过客相寻 ":
                        sub_subject = "过客相寻"
                if i == 81704:
                    if sub_subject == "次韵答舒教授观余所藏墨 ":
                        sub_subject = "次韵答舒教授观余所藏墨"
                if i == 103942:
                    if sub_subject == "分布计算环境  ":
                        sub_subject = "分布计算环境"
                if i == 145741:
                    if sub_subject == "上海正午1":
                        sub_subject = "上海正午"
                if i == 21817:
                    if sub_subject == "蜘蛛侠1":
                        sub_subject = "蜘蛛侠"
                if i == 83569:
                    if sub_subject == " 海滩的一天 ":
                        sub_subject = "海滩的一天"
                if i == 147032:
                    if sub_subject == "野蛮审判 ":
                        sub_subject = "野蛮审判"
                if i == 592:
                    if sub_subject == "Rafael Tol":
                        sub_subject = "Rafael Tolói"
                if i == 2093:
                    if sub_subject == "理发店3 ":
                        sub_subject = "理发店3"
                if i == 2715:
                    if sub_subject == "朱子语类大全）140卷，即今通行本《朱子语类 ":
                        sub_subject = "朱子语类大全）140卷，即今通行本《朱子语类"
                if i == 11217:
                    if sub_subject == "“宠”妃 ":
                        sub_subject = "“宠”妃"
                if i == 14372:
                    if sub_subject == "1CN5A科技广场":
                        sub_subject = "CN5A科技广场"
                if i == 16692:
                    if sub_subject == "看不见的TA之时间裂缝 ":
                        sub_subject = "看不见的TA之时间裂缝"
                if i == 17146:
                    if sub_subject == " 摘星之旅":
                        sub_subject = "摘星之旅"
                if i == 18161:
                    if sub_subject == "色·戒 ":
                        sub_subject = "色·戒"
                if i == 20083:
                    if sub_subject == "趣头条邀请码A8133后接0868 ":
                        sub_subject = "趣头条邀请码A8133后接0868"
                if i == 20961:
                    if sub_subject == "":
                        continue
                if i == 22315:
                    if sub_subject == " 山楂树 ":
                        sub_subject = "山楂树"
                    if sub_subject == " 星星之火":
                        sub_subject = "星星之火"
                if i == 26886:
                    if sub_subject == "特殊案件专案组TEN":
                        sub_subject = "特殊案件专案组TEN2"
                if i == 28701:
                    if sub_subject == "金针诗格 ":
                        sub_subject = "金针诗格"
                if i == 30613:
                    if sub_subject == "胭脂\xa0":
                        sub_subject = "胭脂"
                if i == 37817:
                    if sub_subject == " 夕阳毒·痴情未央":
                        sub_subject = "夕阳毒·痴情未央"
                if i == 51557:
                    if sub_subject == "The Beatles Code":
                        sub_subject = "The Beatles Code2"
                if i == 56961:
                    if sub_subject == "红色赞美诗 /Még kér a nép ":
                        sub_subject = "红色赞美诗 /Még kér a nép"
                if i == 63858:
                    if sub_subject == "银翼杀手2":
                        sub_subject = "银翼杀手2049"
                if i == 65929:
                    if sub_subject == "我和美少妇的秘密 ":
                        sub_subject = "我和美少妇的秘密"
                if i == 66911:
                    if sub_subject == "四库全书总目提要·卷一百四十·子部五十 ":
                        sub_subject = "四库全书总目提要·卷一百四十·子部五十"
                if i == 69827:
                    if sub_subject == " 赤壁赋 ":
                        sub_subject = "赤壁赋"
                if i == 71879:
                    if sub_subject == "#你是我的姐妹# ":
                        sub_subject = "#你是我的姐妹#"
                if i == 72850:
                    if sub_subject == "复仇者联盟4 ":
                        sub_subject = "复仇者联盟4"
                if i == 90738:
                    if sub_subject == " 平凡的重生日子 ":
                        sub_subject = "平凡的重生日子"
                if i == 90841:
                    if sub_subject == "":
                        continue
                if i == 97087:
                    if sub_subject == "莫吟冷夜寒\xa0\xa0 ":
                        sub_subject = "莫吟冷夜寒"
                if i == 98269:
                    if sub_subject == "12年":
                        sub_subject = "2012年"
                if i == 103942:
                    if sub_subject == "分布计算环境 ":
                        sub_subject = "分布计算环境"
                if i == 111527:
                    if sub_subject == " 中国房地产发展报告( No.2) ":
                        sub_subject = "中国房地产发展报告( No.2)"
                if i == 111546:
                    if sub_subject == " 仙嫁 ":
                        sub_subject = "仙嫁"
                if i == 125359:
                    if sub_subject == " 先知，沙与沫":
                        sub_subject = "先知，沙与沫"
                if i == 128861:
                    if sub_subject == "太阳山\xa0":
                        sub_subject = "太阳山"
                if i == 130653:
                    if sub_subject == "只有医生知道 ":
                        sub_subject = "只有医生知道"

                if i == 133985:
                    if sub_subject == " 猫游记":
                        sub_subject = "猫游记"
                if i == 135885:
                    if sub_subject == "跟我的前妻谈恋爱 ":
                        sub_subject = "跟我的前妻谈恋爱"
                if i == 138193:
                    if sub_subject == "Mojang":
                        sub_subject = "MojangAB"
                if i == 138348:
                    if sub_subject == "帝王攻略 ":
                        sub_subject = "帝王攻略"
                if i == 145532:
                    if sub_subject == "暴力街区1":
                        sub_subject = "暴力街区13"
                if i == 146665:
                    if sub_subject == "小恐龙阿贡 | GON -ゴン- ":
                        sub_subject = "小恐龙阿贡 | GON -ゴン-"
                if i == 149793:
                    if sub_subject == "质量管理 ":
                        sub_subject = "质量管理"
                if i == 150635:
                    if sub_subject == " 花田少年史 ":
                        sub_subject = "花田少年史"
                if i == 150641:
                    if sub_subject == "四喜忧国 ":
                        sub_subject = "四喜忧国"
                if i == 154951:
                    if sub_subject == "快穿女配：反派BOSS有毒 ":
                        sub_subject = "快穿女配：反派BOSS有毒"
                if i == 155929:
                    if sub_subject == "轩辕剑之天之痕 ":
                        sub_subject = "轩辕剑之天之痕"
                if i == 165632:
                    if sub_subject == "省委书记 ":
                        sub_subject = "省委书记"
                if i == 166260:
                    if sub_subject == "失踪2 - Night is coming ":
                        sub_subject = "失踪2 - Night is coming"
                if i == 167430:
                    if sub_subject == " 深圳情":
                        sub_subject = "深圳情"
                if i == 167430:
                    if sub_subject == "夜色阑珊 ":
                        sub_subject = "夜色阑珊"
                if i == 168223:
                    if sub_subject == "":
                        continue
                if i == 168675:
                    if sub_subject == "7":
                        sub_subject = "007"
                if i == 169969:
                    if sub_subject == " 神经漫游者 ":
                        sub_subject = "神经漫游者"
                if i == 170037:
                    if sub_subject == "八方战士 ":
                        sub_subject = "八方战士"
                    if sub_subject == "梦幻空间 ":
                        sub_subject = "梦幻空间"
                    if sub_subject == "孤独战神 ":
                        sub_subject = "孤独战神"
                try:
                    sub_indx = train_text.index(sub_subject)
                except Exception as e:
                    print(i, state, "document")
                    print(train_text, [sub_subject])
                    raise Exception

                if i == 167005:
                    if sub_subject == "Haru":
                        sub_indx = train_text.index("李Haru")+1

                entity_sub = Entity(self.entity2id[sub_subject_type], sub_subject, sub_indx,
                                    sub_indx + len(sub_subject))
                self.entity_max_len = max(self.entity_max_len, len(sub_subject))

                entity_list.append(entity_sub)

                obj_object = spo["object"]["@value"]
                sub_object_type = spo["object_type"]["@value"]

                if i == 153266:
                    if obj_object == "KBS明星发掘大赛冠军":
                        obj_object = "2006KBS明星发掘大赛冠军"
                if i == 103625:
                    if obj_object == "Hito流行音乐奖高音质Hito最潜力女声奖":
                        obj_object = "2014Hito流行音乐奖高音质Hito最潜力女声奖"
                if i == 125381:
                    if obj_object == "詹姆斯卡梅隆\xad":
                        obj_object = "詹姆斯卡梅隆"
                if i == 71744:
                    if obj_object == "5":
                        obj_object = "5,505,640"
                if i == 38639:
                    if obj_object == "81102":
                        obj_object = "081102"
                if i == 5147:
                    if obj_object == "degli Studi di Torino":
                        obj_object = "Università degli Studi di Torino"
                if i == 12081:
                    if obj_object == "0:00":
                        obj_object = "00:00"
                if i == 14323:
                    if obj_object == "810":
                        obj_object = "0810"
                if i == 14515:
                    if obj_object == "80202":
                        obj_object = "080202"
                if i == 27401:
                    if obj_object == " 沈阳师范大学":
                        obj_object = "沈阳师范大学"
                if i == 32696:
                    if obj_object == "810":
                        obj_object = "0810"
                if i == 33646:
                    if obj_object == " 最爱还是你":
                        obj_object = "最爱还是你"
                if i == 35493:
                    if obj_object == "MBC演艺大赏年度STAR奖":
                        obj_object = "2013MBC演艺大赏年度STAR奖"
                if i == 36506:
                    if obj_object == "CNS":
                        obj_object = "CNS1952"
                if i == 40600:
                    if obj_object == "为梦想喝彩 ":
                        obj_object = "为梦想喝彩"
                if i == 42982:
                    if obj_object == "QUT":
                        obj_object = "QUTBBS"
                if i == 44859:
                    if obj_object == "MTV封神榜音乐奖十大人气王":
                        obj_object = "2011MTV封神榜音乐奖十大人气王"
                if i == 53638:
                    if obj_object == "n":
                        obj_object = "N"
                if i == 60695:
                    if obj_object == "esley Chiang":
                        obj_object = "Lesley Chiang"
                if i == 71347:
                    if obj_object == "810":
                        obj_object = "0810"
                if i == 75909:
                    if obj_object == "r":
                        obj_object = "Mark-Paul"
                if i == 81827:
                    if obj_object == "80502":
                        obj_object = "080502"
                if i == 89866:
                    if obj_object == "810":
                        obj_object = "0810"
                if i == 91102:
                    if obj_object == "lica Rivera":
                        obj_object = "Angélica Rivera"
                    if obj_object == "Roberto G":
                        obj_object = "Roberto Gómez"
                if i == 98495:
                    if obj_object == "IGA":
                        obj_object = "IIGA"
                if i == 100194:
                    if obj_object == "n":
                        obj_object = "N"
                if i == 102257:
                    if obj_object == "":
                        obj_object = "2016年10月"
                if i == 108822:
                    if obj_object == "810":
                        obj_object = "0810"
                if i == 116304:
                    if obj_object == " 复旦大学":
                        obj_object = "复旦大学"
                if i == 120178:
                    if obj_object == "810":
                        obj_object = "0810"
                if i == 126309:
                    if obj_object == "Marc Ang":
                        obj_object = "Marc Angélil"
                if i == 129608:
                    if obj_object == "52360":
                        obj_object = "052360"
                if i == 130752:
                    if obj_object == "81102":
                        obj_object = "081102"
                if i == 136049:
                    if obj_object == "50011":
                        obj_object = "050011"
                if i == 148182:
                    if obj_object == " 广州美术学院":
                        obj_object = "广州美术学院"
                if i == 152738:
                    if obj_object == "B1A":
                        obj_object = "B1A4"
                if i == 156697:
                    if obj_object == "Daniel Cebri":
                        obj_object = "Daniel Cebrián"
                    if obj_object == "Jaime Ol":
                        obj_object = "Jaime Olías"
                # if i == 5147:
                #     if obj_object == "Università degli Studi di Torino":
                #         obj_object = "Universitàdegli Studi di Torino"

                try:
                    obj_indx = train_text.index(obj_object)
                except Exception as e:
                    print(e, train_text, [obj_object], i)
                    print(e)
                    raise Exception
                if i == 63086:
                    if obj_object == "800万":
                        obj_indx = train_text.index("#800万")+1
                if i == 62125:
                    if obj_object == "3":
                        obj_indx = train_text.index("口3")+1
                if i == 5665:
                    if obj_object == "KBS":
                        obj_indx = train_text.index("尔KBS") + 1
                if i == 12844:
                    if obj_object == "TVB":
                        obj_indx = train_text.index("TVB 9")
                if i == 14575:
                    if obj_object == "2003年":
                        obj_indx = train_text.index("2003年0")
                if i == 44829:
                    if obj_object == "SINO":
                        obj_indx = train_text.index("SINO)")
                if i == 76094:
                    if obj_object == "210万":
                        obj_indx = train_text.index("210万，累计18")
                if i == 80363:
                    if obj_object == "335万":
                        obj_indx = train_text.index("335万 #分歧者")
                    if obj_object == "8万":
                        obj_indx = train_text.index("8万 #性上瘾")
                if i == 81414:
                    if obj_object == "6亿":
                        obj_indx = train_text.index("者6亿")+1
                if i == 110748:
                    if obj_object == "GC":
                        obj_indx = train_text.index("GC）")
                if i == 114995:
                    if obj_object == "ST":
                        obj_indx = train_text.index("ST（")
                if i == 157051:
                    if obj_object == "120万":
                        obj_indx = train_text.index("120万，累计3")

                entity_obj = Entity(self.entity2id[sub_object_type], obj_object, obj_indx,
                                    obj_indx + len(obj_object))
                self.entity_max_len = max(self.entity_max_len, len(obj_object))
                entity_list.append(entity_obj)

                predicate_type = spo["predicate"]
                sub_relation = Relation(self.relation2id[predicate_type], entity_sub, entity_obj)
                relation_list.append(sub_relation)
            if state:
                continue
            doc = Document(i, train_text, train_text_id, entity_list, relation_list)
            self.documents.append(doc)

        self.test_documents = []
        for i, test_data in enumerate(test_data_list):
            test_text = test_data["text"]
            self.max_seq_len = max(self.max_seq_len, len(test_text))
            test_text_id = []
            for tt in test_text:
                test_text_id.append(self.char2id.get(tt, 1))
            entity_list = []
            relation_list = []
            doc = Document(i, test_text, test_text_id, entity_list, relation_list)
            self.test_documents.append(doc)
        self.dev_data_list = []
        self.dev_documents = []
        for i, train_data in enumerate(dev_data_list):

            self.data_len += 1
            self.dev_data_list.append(train_data)
            self.max_seq_len = max(self.max_seq_len, len(train_data["text"]))
            train_text = train_data["text"]
            train_text_id = []
            for tt in train_text:
                if tt not in self.char2id:
                    self.char2id[tt] = len(self.char2id)
                train_text_id.append(self.char2id[tt])
                # train_text_id.append(self.char2id.get(tt, self.char2id["<unk>"]))
            if use_word_feature:
                for tword in jieba.cut(train_text):
                    if tword not in self.word2id:
                        self.word2id[tword] = len(self.word2id)

            spo_list = train_data["spo_list"]

            entity_list = []
            relation_list = []

            state = 0
            for spo in spo_list:
                sub_subject = spo["subject"]
                sub_subject_type = spo["subject_type"]
                try:
                    sub_indx = train_text.index(sub_subject)
                except Exception as e:
                    # print(e, train_text, sub_subject)
                    state = 1
                    print(e)
                    break

                entity_sub = Entity(self.entity2id[sub_subject_type], sub_subject, sub_indx,
                                    sub_indx + len(sub_subject))
                self.entity_max_len = max(self.entity_max_len, len(sub_subject))

                entity_list.append(entity_sub)

                sub_object = spo["object"]["@value"]
                sub_object_type = spo["object_type"]["@value"]
                try:
                    obj_indx = train_text.index(sub_object)
                except Exception as e:
                    # print(e, train_text, sub_object)
                    state = 1
                    print(e)
                    break
                entity_obj = Entity(self.entity2id[sub_object_type], sub_object, obj_indx,
                                    obj_indx + len(sub_object))
                self.entity_max_len = max(self.entity_max_len, len(sub_object))
                entity_list.append(entity_obj)

                predicate_type = spo["predicate"]
                sub_relation = Relation(self.relation2id[predicate_type], entity_sub, entity_obj)
                relation_list.append(sub_relation)
            if state:
                continue
            doc = Document(i, train_text, train_text_id, entity_list, relation_list)
            self.dev_documents.append(doc)

    def eval_metrics(self, input_pres: list):
        assert len(self.dev_data_list) == len(input_pres)
        p_count = 0.0
        d_count = 0.0
        hit = 0.0

        for i, doc in enumerate(self.dev_data_list):
            doc_res = input_pres[i]
            p_count += len(doc_res["spo_list"])
            d_count += len(doc["spo_list"])

            spo_set = set()
            for spo in doc_res["spo_list"]:
                spo_set.add((spo["subject"], spo["subject_type"], spo["object"]["@value"], spo["object_type"]["@value"], spo["predicate"]))

            for doc_spo in doc.relation_list:
                d_spo = (doc_spo["subject"], doc_spo["subject_type"], doc_spo["object"]["@value"], doc_spo["object_type"]["@value"], doc_spo["predicate"])
                if d_spo in spo_set:
                    hit += 1

        return {
            "hit": hit,
            "d_count": d_count,
            "p_count": p_count,
            "recall": (hit + 1e-8) / (d_count + 1e-3),
            "precision": (hit + 1e-8) / (p_count + 1e-3)
        }


class Argument(object):

    def __init__(self, input_argument, input_role_id, input_role, input_start_index):

        self._argument = input_argument
        self._role_id = input_role_id
        self._role = input_role
        self._start = input_start_index

        self.is_enum = 0
        self.enum_items = []

    @property
    def argument(self):
        return self._argument

    @property
    def role_id(self):
        return self._role_id

    @property
    def role(self):
        return self._role

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, input_start):
        self._start = input_start


class Event(object):
    def __init__(self, input_id, input_trigger, input_trigger_start):
        self._id = input_id
        self._trigger = input_trigger
        self._trigger_start = input_trigger_start
        self._arguments = []

    @property
    def id(self):
        return self._id

    @property
    def trigger(self):
        return self._trigger

    @property
    def trigger_start(self):
        return self._trigger_start

    @property
    def arguments(self):
        return self._arguments

    def add_argument(self, input_argument):
        self._arguments.append(input_argument)


class EventDocument(object):
    def __init__(self, input_id, input_text, input_text_id, input_title=None, input_title_id=None):
        self._id = input_id
        self._text = input_text
        self._text_id = input_text_id
        self._title = input_title
        self._title_id = input_title_id
        self._sentence = []
        self._event_list = []

    @property
    def id(self):
        return self._id

    @property
    def text(self):
        return self._text

    @property
    def text_id(self):
        return self._text_id

    @property
    def title_id(self):
        return self._title_id

    @property
    def event_list(self):
        return self._event_list

    @property
    def title(self):
        return self._title

    def add_event(self, input_event):
        self._event_list.append(input_event)


class LoaderBaiduDueeV1(object):
    """
    DuEE1.0是百度发布的中文事件抽取数据集，包含65个事件类型的1.7万个具有事件信息的句子（2万个事件）。
    事件类型根据百度风云榜的热点榜单选取确定，具有较强的代表性。65个事件类型中不仅包含「结婚」、「辞职」、「地震」
    等传统事件抽取评测中常见的事件类型，还包含了「点赞」等极具时代特征的事件类型。具体的事件类型及对应角色见表3。
    数据集中的句子来自百度信息流资讯文本，相比传统的新闻资讯，文本表达自由度更高，事件抽取的难度也更大。
     DuEE1.0 is a Chinese event extraction dataset released by Baidu,
     which consists of 17,000 sentences containing 20,000 events of 65 event types.
     The event types are selected and determined according to the hot search board of Baidu,
     including not only traditional types such as 「marriage」, 「resignation」, and 「earthquake」,
     but also types with bright characteristic of times, such as 「like」.
     The sentences are extracted from Baijiahao news,
     which have freer expression styles compared to traditional news,
     making the extraction task more challenging.
    """

    def __init__(self, data_path):
        schema_path = data_path + "\\duee_schema\\duee_event_schema.json"

        schema_data = load_json_line_data(schema_path)

        self.event2id = {
            "none_event": 0
        }
        self.argument_role2id = {
            "$unk$": 0
        }
        self.event2argument = dict()
        role_enum = dict()
        for schema in schema_data:
            event_type = schema["event_type"]
            if event_type not in self.event2id:
                self.event2id[event_type] = len(self.event2id)
            event_argument = []

            for role in schema["role_list"]:
                if role["role"] not in self.argument_role2id:
                    self.argument_role2id[role["role"]] = len(self.argument_role2id)
                event_argument.append(role["role"])
                if "enum_items" in role:
                    role_enum[role["role"]] = role["enum_items"]
            self.event2argument[self.event2id[event_type]] = event_argument

        self.id2event = {v: k for k, v in self.event2id.items()}
        self.id2argument = {v: k for k, v in self.argument_role2id.items()}

        train_path = data_path + "\\duee_train.json\\duee_train.json"
        dev_path = data_path + "\\duee_dev.json\\duee_dev.json"
        test_path = data_path + "\\duee_test1.json\\duee_test1.json"

        self.documents = []
        self.char2id = {
            "$pad$": 0,
            "$unk$": 1
        }

        train_data = load_json_line_data(train_path)
        for i, sub_train_data in enumerate(train_data):

            text = sub_train_data["text"]
            text_id = []

            for char in text:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                text_id.append(self.char2id[char])

            sub_doc = EventDocument(i, text, text_id)
            for sub_event in sub_train_data["event_list"]:
                event_id = self.event2id[sub_event["event_type"]]
                sub_trigger = sub_event["trigger"]
                sub_trigger_start_index = sub_event["trigger_start_index"]

                event = Event(event_id, sub_trigger, sub_trigger_start_index)

                for sub_argument in sub_event["arguments"]:
                    sub_arg_index = sub_argument["argument_start_index"]
                    sub_arg_role = sub_argument["role"]
                    sub_arg_value = sub_argument["argument"]

                    argument = Argument(sub_arg_value, self.argument_role2id[sub_arg_role], sub_arg_role, sub_arg_index)
                    event.add_argument(argument)
                sub_doc.add_event(event)
            self.documents.append(sub_doc)

        dev_data = load_json_line_data(dev_path)
        self.dev_documents = []
        for i, sub_dev_data in enumerate(dev_data):
            text = sub_dev_data["text"]
            text_id = []

            for char in text:
                text_id.append(self.char2id.get(char, self.char2id["$unk$"]))

            sub_doc = EventDocument(i, text, text_id)
            for sub_event in sub_dev_data["event_list"]:
                event_id = self.event2id[sub_event["event_type"]]
                sub_trigger = sub_event["trigger"]
                sub_trigger_start_index = sub_event["trigger_start_index"]

                event = Event(event_id, sub_trigger, sub_trigger_start_index)

                for sub_argument in sub_event["arguments"]:
                    sub_arg_index = sub_argument["argument_start_index"]
                    sub_arg_role = sub_argument["role"]
                    sub_arg_value = sub_argument["argument"]

                    argument = Argument(sub_arg_value, self.argument_role2id[sub_arg_role], sub_arg_role, sub_arg_index)
                    event.add_argument(argument)
                sub_doc.add_event(event)
            self.dev_documents.append(sub_doc)

        test_data = load_json_line_data(test_path)
        self.test_document = []
        for i, sub_test_data in enumerate(test_data):

            text = sub_test_data["text"]
            text_id = []

            for char in text:
                text_id.append(self.char2id.get(char, self.char2id["$unk$"]))
            sub_doc = EventDocument(sub_test_data["id"], text, text_id)

            self.test_document.append(sub_doc)


class LoaderBaiduDueeFin(object):
    """
    DuEE-fin是百度最新发布的金融领域篇章级事件抽取数据集，包含13个事件类型的1.17万个篇章，
    同时存在部分非目标篇章作为负样例。事件类型来源于常见的金融事件，具体的事件类型及对应角色见表4。
    数据集中的篇章来自金融领域的新闻和公告，覆盖了真实场景中诸多问题。
    """

    def __init__(self, data_path):
        schema_path = data_path + "\\duee_fin_schema\\duee_fin_event_schema.json"
        schema_data = load_json_line_data(schema_path)

        self.event2id = {
            "none_event": 0
        }
        self.argument_role2id = {
            "$unk$": 0
        }

        self.event_argument2id = {
            "pad": 0
        }

        self.event2argument = {0: dict()}
        role_enum = dict()
        for schema in schema_data:
            event_type = schema["event_type"]
            if event_type not in self.event2id:
                self.event2id[event_type] = len(self.event2id)
            self.event2argument.setdefault(self.event2id[event_type], dict())

            for role in schema["role_list"]:
                if role["role"] not in self.argument_role2id:
                    self.argument_role2id[role["role"]] = len(self.argument_role2id)
                self.event2argument[self.event2id[event_type]][self.argument_role2id[role["role"]]] = len(self.event2argument[self.event2id[event_type]])

                if "enum_items" in role:
                    role_enum[role["role"]] = role["enum_items"]
                event_key = "{}_{}".format(event_type, role["role"])
                if event_key not in self.event_argument2id:
                    self.event_argument2id[event_key] = len(self.event_argument2id)
        self.id2event = {v: k for k, v in self.event2id.items()}
        self.id2argument_role = {v: k for k, v in self.argument_role2id.items()}

        train_path = data_path + "\\duee_fin_train.json\\duee_fin_train.json"
        test_path = data_path + "\\duee_fin_test1.json\\duee_fin_test1.json"
        dev_path = data_path + "\\duee_fin_dev.json\\duee_fin_dev.json"

        self.documents = []
        self.char2id = {
            "$pad$": 0,
            "$unk$": 1
        }

        train_data = load_json_line_data(train_path)
        for i, sub_train_data in enumerate(train_data):

            text = sub_train_data["text"]
            title = sub_train_data["title"]
            doc_id = sub_train_data["id"]
            text_id = []
            title_id = []

            for char in text:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                text_id.append(self.char2id[char])

            for char in title:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                title_id.append(self.char2id[char])

            sub_doc = EventDocument(doc_id, text, text_id, title, title_id)
            for sub_event in sub_train_data.get("event_list", []):
                event_id = self.event2id[sub_event["event_type"]]
                sub_trigger = sub_event["trigger"]
                sub_trigger_start_index = -1

                event = Event(event_id, sub_trigger, sub_trigger_start_index)

                for sub_argument in sub_event["arguments"]:
                    # sub_arg_index = text.index(sub_argument["argument"])
                    # try:
                    #     sub_arg_index = text.index(sub_argument["argument"])
                    # except Exception:
                    #     print(sub_argument)
                    #     continue
                    sub_arg_index = -1
                    sub_arg_role = sub_argument["role"]
                    sub_arg_value = sub_argument["argument"]

                    # todo 修改数据
                    if doc_id == "36ce324c5c05bae92f02e78a6ad8d40a" and sub_arg_role == "回购完成时间":
                        sub_arg_value = "2020年第一次临时股东大会审议通过本次回购方案之日起12个月内"
                    if doc_id == "da8f29a5ce27036464fbd06ac3628c8b" and sub_arg_value == "Himalaya\nCapital":
                        sub_arg_value = "Himalaya"
                    if doc_id == "68e59cc8d48a01bc473c17714e44649e":
                        if sub_arg_value == "2015 年":
                            sub_arg_value = "2015  年"
                        if sub_arg_value == "2016 年":
                            sub_arg_value = "2016  年"
                        if sub_arg_value == "约 11 亿元":
                            sub_arg_value = "约  11  亿元"
                    if doc_id == "0cf76bda7c36d1364d533824bb846731":
                        if sub_arg_value == "Wondery 表情":
                            sub_arg_value = "Wondery"
                    if doc_id == "cdfdb1e7b17256dc19d1b61d9ed8e83f":
                        if sub_arg_value == "荣盛建设工程\n有限公司":
                            sub_arg_value = "荣盛建设工程"

                    argument = Argument(sub_arg_value, self.argument_role2id[sub_arg_role], sub_arg_role, sub_arg_index)
                    if sub_arg_role in role_enum:
                        argument.is_enum = 1
                        argument.enum_items = role_enum[sub_arg_role]

                    event.add_argument(argument)
                sub_doc.add_event(event)
            self.documents.append(sub_doc)

        self.dev_documents = []
        dev_data = load_json_line_data(dev_path)
        for i, sub_dev_data in enumerate(dev_data):
            text = sub_dev_data["text"]
            title = sub_dev_data["title"]
            doc_id = sub_dev_data["id"]
            text_id = []
            title_id = []

            for char in text:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                text_id.append(self.char2id[char])

            for char in title:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
                title_id.append(self.char2id[char])

            sub_doc = EventDocument(doc_id, text, text_id, title, title_id)
            for sub_event in sub_dev_data.get("event_list", []):
                event_id = self.event2id[sub_event["event_type"]]
                sub_trigger = sub_event["trigger"]
                sub_trigger_start_index = -1

                event = Event(event_id, sub_trigger, sub_trigger_start_index)

                for sub_argument in sub_event["arguments"]:
                    # sub_arg_index = text.index(sub_argument["argument"])
                    # try:
                    #     sub_arg_index = text.index(sub_argument["argument"])
                    # except Exception:
                    #     print(sub_argument)
                    #     continue
                    sub_arg_index = -1
                    sub_arg_role = sub_argument["role"]
                    sub_arg_value = sub_argument["argument"]
                    #
                    # # todo 修改数据
                    # if doc_id == "36ce324c5c05bae92f02e78a6ad8d40a" and sub_arg_role == "回购完成时间":
                    #     sub_arg_value = "2020年第一次临时股东大会审议通过本次回购方案之日起12个月内"
                    # if doc_id == "da8f29a5ce27036464fbd06ac3628c8b" and sub_arg_value == "Himalaya\nCapital":
                    #     sub_arg_value = "Himalaya"
                    # if doc_id == "68e59cc8d48a01bc473c17714e44649e":
                    #     if sub_arg_value == "2015 年":
                    #         sub_arg_value = "2015  年"
                    #     if sub_arg_value == "2016 年":
                    #         sub_arg_value = "2016  年"
                    #     if sub_arg_value == "约 11 亿元":
                    #         sub_arg_value = "约  11  亿元"
                    # if doc_id == "0cf76bda7c36d1364d533824bb846731":
                    #     if sub_arg_value == "Wondery 表情":
                    #         sub_arg_value = "Wondery"
                    # if doc_id == "cdfdb1e7b17256dc19d1b61d9ed8e83f":
                    #     if sub_arg_value == "荣盛建设工程\n有限公司":
                    #         sub_arg_value = "荣盛建设工程"

                    argument = Argument(sub_arg_value, self.argument_role2id[sub_arg_role], sub_arg_role, sub_arg_index)
                    if sub_arg_role in role_enum:
                        argument.is_enum = 1
                        argument.enum_items = role_enum[sub_arg_role]

                    event.add_argument(argument)
                sub_doc.add_event(event)
            self.dev_documents.append(sub_doc)


class QaPair(object):

    def __init__(self, input_qa_id, input_qa_type, input_q, input_a, input_start, is_impossible=False):
        self._id = input_qa_id
        self._type = input_qa_type
        self._q = input_q
        self._a = input_a
        self._start = input_start
        self._is_impossible = is_impossible

    @property
    def q(self):
        return self._q

    @property
    def a(self):
        return self._a

    @property
    def start(self):
        return self._start



class QaDocument(object):
    def __init__(self, input_id, input_title, input_context):
        self._id = input_id
        self._title = input_title
        self._context = input_context
        self._qa_list = []

    @property
    def id(self):
        return self._id

    @property
    def title(self):
        return self._title

    @property
    def context(self):
        return self._context

    @property
    def qa_list(self):
        return self._qa_list

    def add_qa_item(self, qa):
        self._qa_list.append(qa)


class LoaderDuReaderChecklist(object):

    def __init__(self, data_path):
        self.train_path = data_path+"\dureader_checklist.dataset\dataset\\train.json"
        self.dev_path = data_path+"\dureader_checklist.dataset\dataset\\dev.json"

        train_data = json.load(open(self.train_path, encoding="utf-8"))
        train_data_list = train_data["data"][0]["paragraphs"]

        test_data = json.load(open(self.dev_path, encoding="utf-8"))
        test_data_list = test_data["data"][0]["paragraphs"]

        self.char2id = {"pad": 0, "unk": 1}
        self.documents = []
        for i, paragraph in enumerate(train_data_list):
            context = paragraph["context"]
            title = paragraph["title"]

            doc = QaDocument(i, title, context)

            for char in title:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
            for char in context:
                if char not in self.char2id:
                    self.char2id[char] = len(self.char2id)
            # print(paragraph)
            for qa in paragraph["qas"]:
                question = qa["question"]
                for char in question:
                    if char not in self.char2id:
                        self.char2id[char] = len(self.char2id)
                qa_item = QaPair(qa["id"], qa["type"], question,
                                 qa["answers"][0]["text"], qa["answers"][0]["answer_start"], qa["is_impossible"])
                doc.add_qa_item(qa_item)
            self.documents.append(doc)
        self.dev_documents = []
        for i, paragraph in enumerate(test_data_list):
            context = paragraph["context"]
            title = paragraph["title"]

            doc = QaDocument(i, title, context)
            for qa in paragraph["qas"]:
                question = qa["question"]
                for char in question:
                    if char not in self.char2id:
                        self.char2id[char] = len(self.char2id)
                qa_item = QaPair(qa["id"], qa["type"], question,
                                 qa["answers"][0]["text"], qa["answers"][0]["answer_start"], qa["is_impossible"])
                doc.add_qa_item(qa_item)
            self.dev_documents.append(doc)


class LoaderBaiduDialogV1(object):

    def __init__(self, data_path):
        self.train_path = data_path+"\\Dialog_sample\Dialog_sample\\"


class BaseDataIterator(object):

    def __init__(self, input_loader):
        self.data_loader = input_loader
        self.use_random = True

    def single_doc_processor(self, doc: Document):
        pass

    def padding_batch_data(self, input_batch_data):
        pass

    def train_iter(self, input_batch_num):
        c_batch_data = []
        rg_idxs = np.arange(0, len(self.data_loader.documents))
        np.random.shuffle(rg_idxs)
        for doc_i in rg_idxs:
            doc = self.data_loader.documents[doc_i]
        # for doc in self.data_loader.documents:
            c_batch_data.append(self.single_doc_processor(doc))
            if len(c_batch_data) == input_batch_num:
                yield self.padding_batch_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_batch_data(c_batch_data)

    def data_iter(self, input_batch_num, train=None):
        if train:
            doc_list = self.data_loader.documents
        else:
            doc_list = self.data_loader.test_documents
        c_batch_data = []
        for doc in doc_list:
            c_batch_data.append(self.single_doc_processor(doc))
            if len(c_batch_data) == input_batch_num:
                yield self.padding_batch_data(c_batch_data)
                c_batch_data = []
        if c_batch_data:
            yield self.padding_batch_data(c_batch_data)
