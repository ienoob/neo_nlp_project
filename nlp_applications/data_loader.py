#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/1/29 23:13
    @Author  : jack.li
    @Site    : 
    @File    : data_loader.py

"""
import json

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

        self.labels = ["B-AGE", "I-AGE", "B-ANGLE", "I-ANGLE", "B-AREA", "I-AREA", "B-CAPACTITY", "I-CAPACTITY",
                       "B-DATE", "I-DATE", "B-DECIMAL", "I-DECIMAL", "B-DURATION", "I-DURATION",
                       "B-FRACTION", "I-FRACTION", "B-FREQUENCY", "I-FREQUENCY",  "B-INTEGER", "I-INTEGER",
                       "B-LENGTH", "I-LENGTH", "B-LOCATION", "I-LOCATION", "B-MEASURE", "I-MEASURE",
                       "B-MONEY", "I-MONEY", "B-ORDINAL", "I-ORDINAL", "B-ORGANIZATION", "I-ORGANIZATION",
                       "B-PERCENT", "I-PERCENT",  "B-PERSON", "I-PERSON", "B-PHONE", "I-PHONE",
                       "B-POSTALCODE", "I-POSTALCODE", "B-RATE", "I-RATE", "B-SPEED", "I-SPEED",
                       "B-TEMPERATURE", "I-TEMPERATURE", "B-TIME", "I-TIME", "B-WEIGHT", "I-WEIGHT",
                       "B-WWW", "I-WWW", "O"]

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

    def __init__(self, input_id, input_sub, input_obj):
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
    def __init__(self, data_path):
        self.schema_path = data_path + "//duie_schema//duie_schema.json"
        self.train_path = data_path + "//duie_train.json//duie_train.json"
        self.dev_path = data_path + "//duie_dev.json//duie_dev.json"
        self.test_path = data_path + "//duie_test1.json//duie_test1.json"

        schema_data_list = load_json_line_data(self.schema_path)
        train_data_list = load_json_line_data(self.train_path)
        dev_data_list = load_json_line_data(self.dev_path)
        test_data_list = load_json_line_data(self.test_path)

        self.data_len = 0
        self.entity_max_len = 0
        self.relation2id = dict()
        self.subject2id = dict()
        self.object2id = dict()
        self.entity2id = dict()

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

        self.id2entity = {v:k for k, v in self.entity2id.items()}
        self.id2relation = {v:k for k, v in self.relation2id.items()}

        self.char2id = {
            "<pad>": 0,
            "<unk>": 1
        }

        self.documents = []
        for i, train_data in enumerate(train_data_list):

            self.data_len += 1
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
            self.documents.append(doc)

        self.test_documents = []
        for i, test_data in enumerate(test_data_list):
            test_text = test_data["text"]
            test_text_id = []
            for tt in test_text:
                test_text_id.append(self.char2id.get(tt, 1))
            entity_list = []
            relation_list = []
            doc = Document(i, test_text, test_text_id, entity_list, relation_list)
            self.test_documents.append(doc)


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
        role_enum = dict()
        for schema in schema_data:
            event_type = schema["event_type"]
            if event_type not in self.event2id:
                self.event2id[event_type] = len(self.event2id)
            for role in schema["role_list"]:
                if role["role"] not in self.argument_role2id:
                    self.argument_role2id[role["role"]] = len(self.argument_role2id)
                if "enum_items" in role:
                    role_enum[role["role"]] = role["enum_items"]

        self.id2event = {v: k for k, v in self.event2id.items()}
        self.id2argument = {v: k for k, v in self.argument_role2id.items()}

        train_path = data_path + "\\duee_train.json\\duee_train.json"
        dev_path = data_path + "\\duee_dev.json\\duee_dev.json"
        test_path = data_path + "\\duee_test1.json\\duee_test1.json"

        self.document = []
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
            self.document.append(sub_doc)

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
        self.id2event = {v:k for k,v in self.event2id.items()}

        train_path = data_path + "\\duee_fin_train.json\\duee_fin_train.json"

        self.document = []
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
            self.document.append(sub_doc)


class QaPair(object):

    def __init__(self, input_qa_id, input_qa_type, input_qa):
        self._id = input_qa_id
        self._type = input_qa_type
        self._qa = input_qa


class QaDocument(object):
    def __init__(self, input_title, input_context):
        self._title = input_title
        self._context = input_context
        self._qa_list = []

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
        print(len(train_data_list))

class LoaderBaiduDialogV1(object):

    def __init__(self, data_path):
        self.train_path = data_path+"\\Dialog_sample\Dialog_sample\\"

