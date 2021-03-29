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


class Entity(object):

    def __init__(self, input_id, input_text, input_start, input_end):
        self._id = input_id
        self._entity_text = input_text
        self.size = input_end-input_start
        self._start = input_start
        self._end = input_end


class Relation(object):

    def __init__(self, input_id, input_sub, input_obj):
        self._id = input_id
        self._relation_sub = input_sub
        self._relation_obj = input_obj


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
















