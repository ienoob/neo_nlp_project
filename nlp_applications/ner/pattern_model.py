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
import jieba.posseg as pseg
import numpy as np
from pyhanlp import HanLP
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
# word_embed = {}
logger.info("word2vec is load")

# for schema in schema_data:
#     print(schema)


# 严格结果， 要求完成重合
def hard_score_res_v2(real_data, predict_data):
    assert len(real_data) == len(predict_data)

    score = 0.0
    d_count = 0
    p_count = 0
    for i, rd in enumerate(real_data):
        if len(predict_data[i]) is 0:
            continue
        p_count += len(predict_data[i])
        d_count += len(real_data[i])

        for rd_ele in rd:
            if rd_ele in predict_data[i]:
                score += 1
    matrix = {
        "score": score,
        "p_count": p_count,
        "d_count": d_count,
        "recall": (score+1e-8)/(d_count+1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }
    matrix["f1_value"] = 2*matrix["recall"]*matrix["precision"]/(matrix["recall"]+matrix["precision"])

    return matrix


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


def soft_score_res_v2(real_data, predict_data):
    assert len(real_data) == len(predict_data)
    score = 0.0
    d_count = 0
    p_count = 0

    def tiny_score(i_real, i_pres):
        r_s_ind = i_real[0]
        r_e_ind = i_real[0] + len(i_real[1])
        p_s_ind = i_pres[0]
        p_e_ind = i_pres[0] + len(i_pres[1])
        if r_s_ind > p_s_ind or r_e_ind < p_e_ind:
            mix_area = 0.0
        else:
            mix_area = min(r_e_ind, p_e_ind) - max(r_s_ind, p_s_ind)

        all_area = max(r_e_ind, p_e_ind) - min(r_s_ind, p_s_ind)

        return mix_area/all_area

    for i, rd in enumerate(real_data):
        pre_res = predict_data[i]
        if len(pre_res) is 0:
            continue
        p_count += len(pre_res)
        d_count += len(rd)
        for rd_ele in rd:
            max_score = 0.0
            for pd_ele in pre_res:
                e_score = tiny_score(rd_ele, pd_ele)
                if e_score > max_score:
                    max_score = e_score
            score += max_score

    matrix = {
        "score": score,
        "p_count": p_count,
        "d_count": d_count,
        "recall": (score + 1e-8) / (d_count + 1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }
    matrix["f1_value"] = 2 * matrix["recall"] * matrix["precision"] / (matrix["recall"] + matrix["precision"])
    return matrix


def sample_data(input_event_type, input_event_role, data_source):
    train_data_list = []
    text_list = []
    label_list = []
    for data in data_source:
        arguments = set()
        for event in data["event_list"]:
            if event["event_type"] != input_event_type:
                continue
            for arg in event["arguments"]:
                if arg["role"] != input_event_role:
                    continue
                train_data_list.append((data["text"], arg["argument_start_index"], arg["argument"]))
                arguments.add((arg["argument_start_index"], arg["argument"]))
        if arguments:
            text_list.append(data["text"])
            label_list.append(arguments)

    return train_data_list, text_list, label_list


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


def dtw(input_list_a, input_list_b):
    a_len = len(input_list_a)
    b_len = len(input_list_b)



# 最长子序列
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

def max_sub_sequence_score(seq1, seq2):
    max_seq_len = max_sub_sequence(seq1, seq2)

    return max_seq_len*1.0/max(len(seq1), len(seq2))


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
    if a_norm == 0 or b_norm == 0:
        return 0.0
    similarity = np.dot(input_a, input_b.T) / (a_norm * b_norm)

    return similarity


tree_shreshold = 0.8


# 正则抽取 所有数据
def extract_info_by_pattern(input_pt, input_text, out_extract_infos, max_len=1<<32):
    try:
        gf = re.finditer(input_pt, input_text, flags=re.S)
        for gfi in gf:
            e_span = gfi.group(1)
            if e_span.strip() == "":
                continue
            if len(e_span) > max_len:
                continue
            if (gfi.start(1), e_span) not in out_extract_infos:
                out_extract_infos[(gfi.start(1), e_span)] = len(out_extract_infos)
    except Exception as e:
        print(input_text, input_pt)
        raise Exception


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
        self.word_embed_feature_v2 = []
        self.word_poss_feature_v3 = []
        self.word_parse_feature_v4 = []

        self.parse2id = dict()

    def negative_choice(self, input_text, positive_span=tuple()):
        """ 随机负采样，原理随机获得和目标词不同的区域作为负样本
        Args:
            input_text:
            positive_span:

        Returns:

        """
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

    def positive_negative_feature(self):
        positive_span = set()
        negative_span = set()

        train_text = []
        train_label = []
        last_text = None
        last_label = set()
        for i, text in enumerate(self.train_text):
            if text == last_text:
                last_label.add(self.train_label[i])
            else:
                if last_text:
                    train_text.append(last_text)
                    train_label.append(last_label)
                last_label = set()
                last_label.add(self.train_label[i])
                last_text = text
        if last_text:
            train_text.append(last_text)
            train_label.append(last_label)

        for i, text in enumerate(train_text):
            extract_infos = dict()
            for pt in self.pattern_list:
                extract_info_by_pattern(text, pt, extract_infos, self.max_len)

            extract_infos_reverse = {v: k for k, v in extract_infos.items()}
            for inx in range(len(extract_infos_reverse)):
                extract = (extract_infos_reverse[inx][0], extract_infos_reverse[inx][1])
                if extract in train_label[i]:
                    positive_span.add(extract[1])
                else:
                    negative_span.add(extract[1])

        return positive_span, negative_span

    # 生成词向量特征，最后采用平均的方式
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

    def generate_feature_v2(self, input_list):
        feature = []
        def inner_f(i_char: str):
            if i_char.isalpha():
                return "[a-zA-Z]"
            elif i_char.isdigit():
                return "\d"
            else:
                return i_char
        for data in input_list:
            inner_feature = [inner_f(d) for d in data]
            feature.append(inner_feature)

        return feature

    # 词性特征
    def generate_feature_v3(self, input_list):
        feature = []

        for sentence in input_list:

            sentence_feature = [[cut.flag for w in cut.word] for cut in pseg.cut(sentence)]
            inner_feature = []
            for sf in sentence_feature:
                inner_feature += sf
            feature.append(inner_feature)

        return feature

    # 句法特征
    def generate_feature_v4(self, input_list):
        feature = []

        for sentence in input_list:

            sentence_feature = [[cut.DEPREL for _ in cut.LEMMA] for cut in HanLP.parseDependency(sentence)]
            inner_feature = []
            for sf in sentence_feature:
                inner_feature += sf
            assert len(sentence) == len(inner_feature)
            feature.append(inner_feature)

        return feature

    # 分类器方式
    def train_core_model(self, input_positive_list, input_negative_list):
        p_feature = self.generate_feature(input_positive_list)
        n_feature = self.generate_feature(input_negative_list)
        p_label = [1 for _ in input_positive_list]
        n_label = [0 for _ in input_negative_list]

        feature = p_feature+n_feature
        label = p_label+n_label

        self.clf.fit(feature, label)

    # 相似方式, 词向量相似度
    def train_similarity_model(self, input_positive_list):
        self.word_embed_feature = self.generate_feature(input_positive_list)

    def train_similarity_v1_model(self, input_positive_list):
        pass

    def train_similarity_model_v2(self, input_positive_list):
        self.word_embed_feature_v2 = self.generate_feature_v2(input_positive_list)

    def train_similarity_model_v3(self, input_positive_list):
        word_feature_v3 = self.generate_feature_v3(input_positive_list)
        self.word_poss_feature_v3 = [word_feature_v3[i][label[0]:label[1]] for i, label in self.train_label]

    def train_similarity_model_v4(self, input_positive_list):
        word_feature_v4 = self.generate_feature_v4(input_positive_list)
        self.word_parse_feature_v4 = [word_feature_v4[i][label[0]:label[1]] for i, label in self.train_label]

    def train_similarity_model_v5(self, input_positive_list):
        pass

    def train_similarity_model_v6(self, input_positive_list):
        self.word_embed_feature = self.generate_feature(input_positive_list)

    # 相似方式，
    def train_similarity_model_all(self):
        # 1 最简单，字一样就行
        word_dict = dict()
        # 2 高级一点，将一些符号转化为一些统一标识。 例如：如果是字母则表示为a-zA-Z, 如果是数字则表示为\d, [\u4e00-\u9fa5] [\x80-\xff] 中文

        # 3 词性

        # 4 句法特征

        # 5 通用ner 特性

        # 6 词向量特征

    # def train_single_pattern_model(self, input_text, input_label_data):
    #     start_indx = input_label_data[0]
    #     end_indx = input_label_data[0] + len(input_label_data[1])
    #     start_context = input_text[:start_indx]
    #     core_data = input_label_data[1]
    #     end_context = input_text[end_indx:]

    def build_pattern_tree(self, input_pattern_start, input_pattern_end, input_match_list, start_len, end_len):
        filter_char = [")", "*", "+", "?", "(", "-", "|"]
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
        inner_not_match_list = []
        inner_match_list = []
        for text_i in input_match_list:
            match_res = re.findall(pattern, self.train_text[text_i], flags=re.S)
            if len(match_res) == 1 and match_res[0] == self.train_label[text_i][1]:
                match_count += 1
                inner_match_list.append(text_i)
            else:
                inner_not_match_list.append(text_i)

        # 这里增加了阈值判断，避免分支过多
        if match_count*1.0/len(input_match_list) >= tree_shreshold:
            self.n_pattern_list.append((pattern, inner_match_list))
            if inner_not_match_list:
                pattern_statis = dict()
                for text_i in inner_not_match_list:
                    start_i = self.train_label[text_i][0]
                    end_i = start_i + len(self.train_label[text_i][1])
                    if start_i - start_len - 1 < 0 and end_i + end_len >= len(self.train_text[text_i]):
                        continue
                    if start_i - start_len - 1 < 0:
                        p_last = input_pattern_end + self.train_text[text_i][end_i + end_len]
                        pattern_statis.setdefault((input_pattern_start, p_last), [])
                        pattern_statis[(input_pattern_start, p_last)].append(text_i)

                    elif end_i + end_len >= len(self.train_text[text_i]):
                        p_first = self.train_text[text_i][start_i - start_len - 1] + input_pattern_start
                        pattern_statis.setdefault((p_first, input_pattern_end), [])
                        pattern_statis[(p_first, input_pattern_end)].append(text_i)
                    else:
                        if start_len == end_len:
                            p_first = self.train_text[text_i][start_i - start_len - 1] + input_pattern_start
                            pattern_statis.setdefault((p_first, input_pattern_end), [])
                            pattern_statis[(p_first, input_pattern_end)].append(text_i)
                        else:
                            p_last = input_pattern_end + self.train_text[text_i][end_i + end_len]
                            pattern_statis.setdefault((input_pattern_start, p_last), [])
                            pattern_statis[(input_pattern_start, p_last)].append(text_i)

                for k, v in pattern_statis.items():
                    sub_start, sub_end = k
                    if len(sub_start) > len(input_pattern_start):
                        self.build_pattern_tree(sub_start, sub_end, v, start_len + 1, end_len)
                    else:
                        self.build_pattern_tree(sub_start, sub_end, v, start_len, end_len + 1)
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

    def check_pattern(self):
        assert len(self.n_pattern_list) > 0
        d = [0]*len(self.train_label)
        for _, m_list in self.n_pattern_list:
            for m in m_list:
                d[m] = 1
        logger.info("data num {0} cover num {1}".format(len(self.train_label), sum(d)))


    def fit(self, input_feature_data, label_datas):
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

        for (start_p, end_p), v in pattern_statis.items():
            if start_p == "^" and end_p == "$":
                continue
            if start_p == "^":
                self.build_pattern_tree(start_p, end_p, v, 0, 1)
            elif end_p == "$":
                self.build_pattern_tree(start_p, end_p, v, 1, 0)
            else:
                self.build_pattern_tree(start_p, end_p, v, 1, 1)
        # 检查模式是否全部覆盖数据
        self.check_pattern()

        pattern_list_value = [(p, len(c)) for p, c in self.n_pattern_list]
        pattern_list_value.sort(key=lambda x: x[1], reverse=True)
        print("pattern_num:", len(pattern_list_value))
        self.pattern_list = [p for p, _ in pattern_list_value]

        # positive_span_value, negative_span_value = self.positive_negative_feature()
        # if len(negative_span_value):
        #     negative_span = list(negative_span_value)
        # self.train_core_model(self.core_list, negative_span)
        self.train_similarity_model(self.core_list)

    def calculate(self, input_str):
        score = 0.0
        for core in self.core_list:
            score += max_sub_sequence(input_str, core)

        return score


    def calculate_word_embed_sim(self, input_span_feature):
        assert len(self.word_embed_feature) > 0
        inner_threshold = 0.999
        filter_span = []
        for iv_ind, span_feature in enumerate(input_span_feature):
            score = max([cosine_distance(span_feature, cmp_embed) for cmp_embed in self.word_embed_feature])
            filter_span.append((iv_ind, score))
            # if score > res_score:
            #     res_score = score
            #     res_ind = iv_ind
        filter_span.sort(key=lambda x: x[1], reverse=True)
        filter_span_score = [(k, v) for k, v in filter_span if v > inner_threshold]
        return filter_span_score[:1]

    def calculate_word_embed_sim_v2(self, input_span_feature):
        assert len(self.word_embed_feature_v2) > 0
        inner_threshold = 0.5
        filter_span = []
        for iv_ind, span_feature in enumerate(input_span_feature):
            score = max([max_sub_sequence_score(span_feature, cmp_embed) for cmp_embed in self.word_embed_feature_v2])
            filter_span.append((iv_ind, score))
            # if score > res_score:
            #     res_score = score
            #     res_ind = iv_ind
        filter_span.sort(key=lambda x: x[1], reverse=True)
        filter_span_score = [(k, v) for k, v in filter_span if v > inner_threshold]
        return filter_span_score[:1]

    def calculate_classifier_score(self, input_span_feature):
        extract_span_feature = self.generate_feature(input_span_feature)
        extract_span_res = self.clf.predict_proba(extract_span_feature)

        filter_span = []
        inner_threshold = 0.8
        for i, span_res in enumerate(extract_span_res[:,1]):

            filter_span.append((i, span_res))
        filter_span.sort(key=lambda x: x[1], reverse=True)
        filter_span_score = [(k, v) for k, v in filter_span if v>inner_threshold]

        if len(filter_span_score):
            return filter_span_score
        else:
            return filter_span[:1]

    def predict(self, input_text):
        predict_res = []
        for text in input_text:
            extract_list = []
            extract_infos = dict()
            for pt in self.pattern_list:
                extract_info_by_pattern(text, pt, extract_infos, self.max_len)

            # extract_infos.sort(key=lambda x: x[2], reverse=True)
            extract_infos_reverse = {v: k for k, v in extract_infos.items()}
            if extract_infos:
                extract_span_list = [e_info[1] for e_info in extract_infos]
                extract_span_feature = self.generate_feature(extract_span_list)
                # extract_span_res = self.calculate_classifier_score(extract_span_list)

                # extract_span_res = self.calculate_word_embed_sim(extract_span_feature)
                extract_span_res = self.calculate_word_embed_sim(extract_span_feature)


                # inx = 0
                # print("max {0} score is {1} ".format(extract_infos_reverse[inx][1], match_score))
                for inx, e_score in extract_span_res:
                    extract = (extract_infos_reverse[inx][0], extract_infos_reverse[inx][1])
                    extract_list.append(extract)

            predict_res.append(extract_list)

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

# 这是用来检查，答案是否在抽取的片段中
def check_if_answer_in(real_ans, pred_ans):
    assert len(real_ans) == len(pred_ans)

    for i, ra in enumerate(real_ans):
        pa = pred_ans[i]
        pa_set = set(pa)
        for ri in ra:
            if ri not in pa_set:
                print(ra, pa)
                raise Exception

import time
import json
if __name__ == "__main__":
    train_eval_info = []
    dev_eval_info = []
    for schema in schema_data:
        event_type = schema["event_type"]
        for role in schema["role_list"]:
            role_value = role["role"]
            # if event_type != "灾害/意外-车祸" or role_value != "地点":
            #     continue

            test_train, test_train_data, test_train_label = sample_data(event_type, role_value, train_data)
            test_eval, test_eval_data, test_eval_label = sample_data(event_type, role_value, eval_data)
            logger.info("event {0} start, role {1} start, train_data {2}".format(event_type, role_value, len(test_train)))
            if len(test_train) == 0:
                continue
            test_train_feature = [d[0] for d in test_train]
            test_for_train_label = [(d[1], d[2]) for d in test_train]

            # test_eval_feature = [d[0] for d in test_eval]
            # test_eval_label = [(d[1], d[2]) for d in test_eval]
            pt = PatternModel()
            pt.fit(test_train_feature, test_for_train_label)
            train_pres = pt.predict(test_train_data)
            eval_pres = pt.predict(test_eval_data)

            # check_if_answer_in(test_train_label, train_pres)
            # logger.info("event ", event_type, " start, role ", role_value, " end")

            train_hard_eval = hard_score_res_v2(test_train_label, train_pres)
            print(train_hard_eval)
            train_soft_eval = soft_score_res_v2(test_train_label, train_pres)

            eval_hard_eval = hard_score_res_v2(test_eval_label, eval_pres)
            print(eval_hard_eval)
            eval_soft_eval = soft_score_res_v2(test_eval_label, eval_pres)

            calculate_result(train_eval_dict, train_hard_eval, train_soft_eval)
            calculate_result(eval_eval_dict, eval_hard_eval, eval_soft_eval)

            train_eval_info.append({"hard": train_hard_eval, "sorf": train_soft_eval})
            dev_eval_info.append({"hard": eval_hard_eval, "sorf": eval_soft_eval})


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

    with open("D:\\tmp\\result_save\\{}".format("train_eval_{}.json".format(int(time.time()))), "w") as f:
        f.write("\n".join([json.dumps(sub_info) for sub_info in train_eval_info]))

    with open("D:\\tmp\\result_save\\{}".format("dev_eval_{}.json".format(int(time.time()))), "w") as f:
        f.write("\n".join([json.dumps(sub_info) for sub_info in dev_eval_info]))


# p_value = hit_count/pred_count
# r_value = hit_count/real_count
# f1_value =
# print(p_value, r_value, f1_value)
