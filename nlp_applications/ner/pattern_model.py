#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/4/25 21:03
    @Author  : jack.li
    @Site    : 
    @File    : pattern_model.py

"""
"""
    这个方法名为模板方法，但是模板是从数据中来，和人工指定模板的思路不同
"""
import re
import json
import logging
import jieba
import numpy as np
from nlp_applications.data_loader import load_json_line_data
from nlp_applications.utils import load_word_vector
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

schema_path = "D:\data\句子级事件抽取\duee_schema\\duee_event_schema.json"
data_path = "D:\data\句子级事件抽取\duee_train.json\\duee_train.json"
eval_data_path = "D:\data\句子级事件抽取\duee_dev.json\\duee_dev.json"
word_embed_path = "D:\\data\\word2vec\\sgns.weibo.char\\sgns.weibo.char"

schema_data = load_json_line_data(schema_path)
train_data = load_json_line_data(data_path)
eval_data = load_json_line_data(eval_data_path)


train_data = list(train_data)
eval_data = list(eval_data)

word_embed = load_word_vector(word_embed_path)
logger.info("word2vec is load")

# for schema in schema_data:
#     print(schema)


# 严格结果， 要求完成重合
def hard_score_res(real_data, predict_data):
    assert len(real_data) == len(predict_data)

    score = 0.0
    d_count = len(real_data)
    p_count = 0
    for i, rd in enumerate(real_data):
        if len(predict_data[i]) is 0:
            continue
        p_count += 1
        if rd[0] == predict_data[i][0] and rd[1] == predict_data[i][1]:
            score += 1

    matrix = {
        "score": score,
        "p_count": p_count,
        "d_count": d_count,
        "recall": (score+1e-8)/(d_count+1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }

    return matrix


# 软评估结果, 不要求完全重合
def soft_score_res(real_data, predict_data):
    assert len(real_data) == len(predict_data)
    score = 0.0
    d_count = len(real_data)
    p_count = 0
    for i, rd in enumerate(real_data):
        pre_res = predict_data[i]
        if len(pre_res) is 0:
            continue
        p_count += 1
        r_s_ind = rd[0]
        r_e_ind = rd[0]+len(rd[1])
        p_s_ind = pre_res[0]
        p_e_ind = pre_res[0]+len(pre_res[1])
        if r_s_ind > p_s_ind or r_e_ind < p_e_ind:
            mix_area = 0.0
        else:
            mix_area = min(r_e_ind, p_e_ind)-max(r_s_ind, p_s_ind)

        all_area = max(r_e_ind, p_e_ind)-min(r_s_ind, p_s_ind)

        score += mix_area/all_area
    matrix = {
        "score": score,
        "p_count": p_count,
        "d_count": d_count,
        "recall": (score + 1e-8) / (d_count + 1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }

    return matrix


def sample_data(input_event_type, input_event_role, data_source):
    data_list = []
    for data in data_source:
        for event in data["event_list"]:
            if event["event_type"] != input_event_type:
                continue
            for arg in event["arguments"]:
                if arg["role"] != input_event_role:
                    continue
                data_list.append((data["text"], arg["argument_start_index"], arg["argument"]))

    return data_list


def bm25(query, doc_list):

    da = []
    d = dict()
    for doc in doc_list:
        di = dict()
        for char in doc:
            d.setdefault(char, 0)
            d[char] += 1
            di.setdefault(char, 0)
            di[char] += 1
        da.append(di)


# Levenshtein
def distance():
    pass


def dtw():
    pass


def max_sub_sequence(seq1, seq2):
    seq1_len = len(seq1)
    seq2_len = len(seq2)
    dp = [[0 for i in range(seq2_len+1)] for j in range(seq1_len+1)]

    for i, s1 in enumerate(seq1):
        for j, s2 in enumerate(seq2):
            if s1 == s2:
                dp[i+1][j+1] = max([dp[i][j]+1, dp[i][j+1], dp[i+1][j]])
            else:
                dp[i+1][j+1] = max([dp[i][j], dp[i][j+1], dp[i+1][j]])

    return dp[-1][-1]


print(max_sub_sequence("abc", "yagbghachje"))


charater_list = [".", "+"]


def single_pattern(input_str):
    pattern = []
    for char in input_str:
        if char.isdigit():
            pattern.append("d")
        elif char.isalpha():
            pattern.append("w")
        elif '\u4e00' <= char <= '\u9fff':
            pattern.append("c")
        else:
            pattern.append(char)

    return pattern


def cosine_distance(input_a, input_b):
    a_norm = np.linalg.norm(input_a)
    b_norm = np.linalg.norm(input_b)
    similarity = np.dot(input_a, input_b.T) / (a_norm * b_norm)

    return similarity


tree_shreshold = 0.8


class PatternModel(object):

    def __init__(self):
        self.pattern_list = []
        self.core_list = []
        self.core_pattern = []
        self.clf = LogisticRegression(solver='lbfgs')
        self.max_len = 0
        self.train_text = []
        self.train_label = []
        self.n_pattern_list = []
        self.word_embed_feature = []

    def negative_choice(self, input_text, positive_span=tuple()):
        dlen = len(input_text)
        si, ei = positive_span

        def match_span(a1, a2, b1, b2):
            if a1 > b2 or b1 > a2:
                return False
            return True
        while True:
            i = np.random.randint(0, dlen)
            j = np.random.randint(0, dlen)
            s = min(i, j)
            e = max(i, j)+1
            if match_span(s, e, si, ei) or input_text[s:e].strip() == "":
                continue
            return input_text[s:e]

    def generate_feature(self, input_list):
        features = []
        for data in input_list:
            feature = None
            p_count = 0
            # 这里试试先分词
            data = jieba.cut(data)
            for char in data:
                if char not in word_embed:
                    char_embed = np.zeros(300)
                else:
                    char_embed = word_embed[char]
                if feature is None:
                    feature = char_embed
                else:
                    feature += char_embed
                p_count += 1
            feature /= p_count
            features.append(feature)
        return features

    # 分类器方式
    def train_core_model(self, input_positive_list, input_negative_list):
        p_feature = self.generate_feature(input_positive_list)
        n_feature = self.generate_feature(input_negative_list)
        p_label = [1 for _ in input_positive_list]
        n_label = [0 for _ in input_negative_list]

        feature = p_feature+n_feature
        label = p_label+n_label

        self.clf.fit(feature, label)

    # 相似方式
    def train_similarity_model(self, input_positive_list):
        self.word_embed_feature = self.generate_feature(input_positive_list)

    # def train_single_pattern_model(self, input_text, input_label_data):
    #     start_indx = input_label_data[0]
    #     end_indx = input_label_data[0] + len(input_label_data[1])
    #     start_context = input_text[:start_indx]
    #     core_data = input_label_data[1]
    #     end_context = input_text[end_indx:]

    def build_pattern_tree(self, input_pattern_start, input_pattern_end, input_match_list, start_len, end_len):
        filter_char = [")", "*", "+", "?", "(", "-"]
        input_pattern_start_n = []
        for p in input_pattern_start:
            if p in filter_char:
                p = "\\{}".format(p)
            input_pattern_start_n.append(p)
        input_pattern_end_n = []
        for p in input_pattern_end:
            if p in filter_char:
                p = "\\{}".format(p)
            input_pattern_end_n.append(p)
        pattern = r"{0}(.+?){1}".format("".join(input_pattern_start_n), "".join(input_pattern_end_n))
        match_count = 0
        for text_i in input_match_list:
            match_res = re.findall(pattern, self.train_text[text_i])
            if len(match_res) == 1 and match_res[0] == self.train_label[text_i][1]:
                match_count += 1

        # 这里增加了阈值判断，避免分支过多
        if match_count*1.0/len(input_match_list) > tree_shreshold:
            self.n_pattern_list.append((pattern, input_match_list))
        else:
            pattern_statis = dict()
            for text_i in input_match_list:
                start_i = self.train_label[text_i][0]
                end_i = start_i + len(self.train_label[text_i][1])
                if start_i-start_len-1 < 0 and end_i+end_len >= len(self.train_text[text_i]):
                    continue
                if start_i-start_len-1 < 0:
                    p_last = input_pattern_end + self.train_text[text_i][end_i + end_len]
                    pattern_statis.setdefault((input_pattern_start, p_last), [])
                    pattern_statis[(input_pattern_start, p_last)].append(text_i)

                elif end_i+end_len >= len(self.train_text[text_i]):
                    p_first = self.train_text[text_i][start_i-start_len-1] + input_pattern_start
                    pattern_statis.setdefault((p_first, input_pattern_end), [])
                    pattern_statis[(p_first, input_pattern_end)].append(text_i)
                else:
                    if start_len == end_len:
                        p_first = self.train_text[text_i][start_i-start_len-1] + input_pattern_start
                        pattern_statis.setdefault((p_first, input_pattern_end), [])
                        pattern_statis[(p_first, input_pattern_end)].append(text_i)
                    else:
                        p_last = input_pattern_end + self.train_text[text_i][end_i + end_len]
                        pattern_statis.setdefault((input_pattern_start, p_last), [])
                        pattern_statis[(input_pattern_start, p_last)].append(text_i)

            for k, v in pattern_statis.items():
                sub_start, sub_end = k
                if len(sub_start) > len(input_pattern_start):
                    self.build_pattern_tree(sub_start, sub_end, v, start_len+1, end_len)
                else:
                    self.build_pattern_tree(sub_start, sub_end, v, start_len, end_len+1)

    def fit(self, input_feature_data, label_datas):
        filter_char = [")", "*", "+", "?", "(", "-"]
        pattern_statis = dict()
        # 负样本
        negative_span = []
        self.train_text = input_feature_data
        self.train_label = label_datas
        for i, text in enumerate(input_feature_data):
            label_data = label_datas[i]
            start_indx = label_data[0]
            end_indx = label_data[0]+len(label_data[1])
            start_context = text[:start_indx]
            core_data = label_data[1]
            self.core_list.append(core_data)
            negative_span.append(self.negative_choice(text, (start_indx, end_indx)))
            end_context = text[end_indx:]

            start_p = start_context[-1] if len(start_context) else "^"
            end_p = end_context[0] if len(end_context) else "$"

            pattern_statis.setdefault((start_p, end_p), [])
            pattern_statis[(start_p, end_p)].append(i)

            self.max_len = max(self.max_len, len(core_data))

        self.train_core_model(self.core_list, negative_span)
        self.train_similarity_model(self.core_list)
        for (start_p, end_p), v in pattern_statis.items():
            if start_p == "^" and end_p == "$":
                continue
            if start_p == "^":
                self.build_pattern_tree(start_p, end_p, v, 0, 1)
            elif end_p == "$":
                self.build_pattern_tree(start_p, end_p, v, 1, 0)
            else:
                self.build_pattern_tree(start_p, end_p, v, 1, 1)



        # pattern_list = [(p[0], p[1], len(c)) for p, c in pattern_statis.items()]
        #
        # pattern_list.sort()
        # pattern_list_value = []
        # last = None
        #
        # for p1, p2, c in pattern_list:
        #     if p1 in filter_char:
        #         p1 = "\\{}".format(p1)
        #     if p2 in filter_char:
        #         p2 = "\\{}".format(p2)
        #     if last is not None:
        #         if last[0] != p1:
        #             pattern_list_value.append(last)
        #             last = (p1, [p2], c)
        #         else:
        #             last = (p1, [p2]+last[1], c+last[2])
        #     else:
        #         last = (p1, [p2], c)
        # if last:
        #     pattern_list_value.append(last)
        #
        # pattern_list_value.sort(key=lambda x: x[2], reverse=True)
        # for p1, p2, _ in pattern_list_value:
        #     pattern = r"{0}(.+?)[{1}]".format(p1, "".join(p2))
        #     self.pattern_list.append(pattern)

        pattern_list_value = [(p, len(c)) for p, c in self.n_pattern_list]
        pattern_list_value.sort(key=lambda x: x[1], reverse=True)
        print("pattern_num:", len(pattern_list_value))
        self.pattern_list = [p for p, _ in pattern_list_value]

    def calculate(self, input_str):
        score = 0.0
        for core in self.core_list:
            score += max_sub_sequence(input_str, core)

        return score

    # def calculate_pattern_score(self, input_str):
    #     score = 0.0
    #     for c_pattern in self.core_pattern:
    #         if re.fullmatch(c_pattern, )

    def calculate_word_embed_sim(self, input_span_feature):
        assert len(self.word_embed_feature) > 0
        res_ind = -1
        res_score = -1
        for iv_ind, span_feature in enumerate(input_span_feature):
            score = max([cosine_distance(span_feature, cmp_embed) for cmp_embed in self.word_embed_feature])
            if score > res_score:
                res_score = score
                res_ind = iv_ind
        return res_ind, res_score

    def predict(self, input_text):
        predict_res = []
        for text in input_text:
            extract = tuple()
            extract_infos = dict()
            for pt in self.pattern_list:
                try:
                    gf = re.finditer(pt, text)
                    for gfi in gf:
                        e_span = gfi.group(1).strip()
                        if e_span == "":
                            continue
                        if len(e_span) > self.max_len:
                            continue
                            # e_span_pattern = single_pattern(e_span)
                        if (gfi.start(1), e_span) not in extract_infos:
                            extract_infos[(gfi.start(1), e_span)] = len(extract_infos)
                    # g = re.search(pt, text)
                    # if g:
                    #     e_span = g.group(1)
                    #     if e_span.strip():
                    #     # e_span_pattern = single_pattern(e_span)
                    #
                    #         extract_infos.append((g.start(1), e_span, 0))
                except Exception as e:
                    print(text, pt)
                    raise Exception

            # extract_infos.sort(key=lambda x: x[2], reverse=True)
            extract_infos_reverse = {v: k for k, v in extract_infos.items()}
            if extract_infos:
                inx = 0
                extract_span_list = [e_info[1] for e_info in extract_infos]
                extract_span_feature = self.generate_feature(extract_span_list)
                # extract_span_res = self.clf.predict_proba(extract_span_feature)
                # inx = np.argmax(extract_span_res[:,1])
                assert len(extract_span_feature) > 0
                inx, match_score = self.calculate_word_embed_sim(extract_span_feature)

                if inx == -1:
                    inx = 0
                print("max {0} score is {1} ".format(extract_infos_reverse[inx][1], match_score))

                extract = (extract_infos_reverse[inx][0], extract_infos_reverse[inx][1])
            predict_res.append(extract)

        return predict_res


hit_count = 0.0
pred_count = 0.0
real_count = 0.0
train_eval_dict = {
    "hard_hit_count": 0,
    "soft_hit_count": 0,
    "hard_pre_count": 0,
    "soft_pre_count": 0,
    "hard_real_count": 0,
    "soft_real_count": 0
}
eval_eval_dict = {
    "hard_hit_count": 0,
    "soft_hit_count": 0,
    "hard_pre_count": 0,
    "soft_pre_count": 0,
    "hard_real_count": 0,
    "soft_real_count": 0
}


def calculate_result(input_eval_dict, hard_eval, soft_eval):
    input_eval_dict["hard_hit_count"] += hard_eval["score"]
    input_eval_dict["hard_pre_count"] += hard_eval["p_count"]
    input_eval_dict["hard_real_count"] += hard_eval["d_count"]
    input_eval_dict["soft_hit_count"] += soft_eval["score"]
    input_eval_dict["soft_pre_count"] += soft_eval["p_count"]
    input_eval_dict["soft_real_count"] += soft_eval["d_count"]


for schema in schema_data:
    event_type = schema["event_type"]
    for role in schema["role_list"]:
        role_value = role["role"]

        test_train = sample_data(event_type, role_value, train_data)
        test_eval = sample_data(event_type, role_value, eval_data)
        print("event ", event_type, " start, role ", role_value, " start, train data, ", len(test_train))
        if len(test_train) == 0:
            continue
        test_train_feature = [d[0] for d in test_train]
        test_train_label = [(d[1],d[2]) for d in test_train]

        test_eval_feature = [d[0] for d in test_eval]
        test_eval_label = [(d[1],d[2]) for d in test_eval]
        pt = PatternModel()
        pt.fit(test_train_feature, test_train_label)
        train_pres = pt.predict(test_train_feature)
        eval_pres = pt.predict(test_eval_feature)

        train_hard_eval = hard_score_res(test_train_label, train_pres)
        train_soft_eval = soft_score_res(test_train_label, train_pres)

        eval_hard_eval = hard_score_res(test_eval_label, eval_pres)
        eval_soft_eval = soft_score_res(test_eval_label, eval_pres)

        calculate_result(train_eval_dict, train_hard_eval, train_soft_eval)
        calculate_result(eval_eval_dict, eval_hard_eval, eval_soft_eval)

train_hard_p_value = train_eval_dict["hard_hit_count"]/train_eval_dict["hard_pre_count"]
train_hard_r_value = train_eval_dict["hard_hit_count"]/train_eval_dict["hard_real_count"]
train_hard_f1_value = 2*train_hard_p_value*train_hard_r_value/(train_hard_p_value+train_hard_r_value)

logger.info("train hard precision: {0} recall: {1} f1_value: {2}".format(train_hard_p_value, train_hard_r_value, train_hard_f1_value))

train_soft_p_value = train_eval_dict["soft_hit_count"]/train_eval_dict["soft_pre_count"]
train_soft_r_value = train_eval_dict["soft_hit_count"]/train_eval_dict["soft_real_count"]
train_soft_f1_value = 2*train_soft_p_value*train_soft_r_value/(train_soft_p_value+train_soft_r_value)

logger.info("train soft precision: {0} recall: {1} f1_value: {2}".format(train_soft_p_value, train_soft_r_value, train_soft_f1_value))

soft_hard_p_value = eval_eval_dict["hard_hit_count"]/eval_eval_dict["hard_pre_count"]
soft_hard_r_value = eval_eval_dict["hard_hit_count"]/eval_eval_dict["hard_real_count"]
soft_hard_f1_value = 2*soft_hard_p_value*soft_hard_r_value/(soft_hard_p_value+soft_hard_r_value)

logger.info("eval hard precision: {0} recall: {1} f1_value: {2}".format(soft_hard_p_value, soft_hard_r_value, soft_hard_f1_value))

soft_soft_p_value = eval_eval_dict["soft_hit_count"]/eval_eval_dict["soft_pre_count"]
soft_soft_r_value = eval_eval_dict["soft_hit_count"]/eval_eval_dict["soft_real_count"]
soft_soft_f1_value = 2*soft_soft_p_value*soft_soft_r_value/(soft_soft_p_value+soft_soft_r_value)

logger.info("eval soft precision: {0} recall: {1} f1_value: {2}".format(soft_soft_p_value, soft_soft_r_value, soft_soft_f1_value))

# p_value = hit_count/pred_count
# r_value = hit_count/real_count
# f1_value =
# print(p_value, r_value, f1_value)
