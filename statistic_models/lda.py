#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    主题模型（LDA Latent Dirichlet Allocation）
    使用gibbs 采样
"""

import logging
import numpy as np
import jieba

logging.basicConfig(level = logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("test info")

doc_file = ["路德宗教改革３００年后，由门德尔松写下的这部交响曲，可以说是第一次用交响曲这种最强有力的体裁来公开表现这一重大主题。",
            "同时，这也是门德尔松个人创作所涉及的最重大主题。",
            "比起贝多芬和舒伯特来，门德尔松在表现重大的社会、人生主题方面显然逊色不少，他主要是个浪漫主义的抒情大师。",
            "第四乐章由两个性格各异的主题交替变奏而成。",
            "第一主题开始时由木管乐器轻轻地奏出，充满了虔诚的崇拜心情，它逐渐地改变音区和配器，直至成为庄严雄浑的圣咏；第二主题则更象一首世俗的合唱进行曲，是真正的群众心底流淌出的欢乐之歌。",
            "一个班级团支部为此组织了主题会，收到了很好的教育效果。"]
alpha = 0.2
beta = 0.5
iter_num = 100


theta = 0.2   # doc -> topic
phi = 0.5  # topic -> word
tassign = 0


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

    def input_words(self, file_str):
        words = jieba.cut(file_str)
        for wd in words:
            self.words.append(wd)
            self.length += 1


class LDA(object):

    def __init__(self, doc_file, big_k, alpha, beta, iter_num):
        self.doc = []
        self.words = []
        for doc in doc_file:
            _doc = Document()
            _doc.input_words(doc)

            for wd in _doc.words:
                if wd in self.words:
                    continue
                self.words.append(wd)

            self.doc.append(_doc)
        self.big_k = big_k
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num

        self.nw = np.zeros((len(self.words), self.big_k))
        self.nwsum = np.zeros(self.big_k)
        self.nd = np.zeros((len(self.doc), self.big_k))
        self.ndsum = np.zeros(len(self.doc))
        self.z = [[0]*len(_doc.words) for _doc in self.doc]


        # init
        for m, doc in enumerate(self.doc):
            for n, wd in enumerate(doc.words):
                topic_index = np.random.randint(0, self.big_k)
                self.z[m][n] = topic_index
                word_id = self.words.index(wd)
                self.nw[word_id][topic_index] += 1
                self.nwsum[topic_index] += 1
                self.nd[m][topic_index] += 1
                self.ndsum[m] += 1

        self.theta = np.zeros((len(self.doc), self.big_k))
        self.phi = np.zeros((self.big_k, len(self.words)))


    def sample(self):
        for iter in range(self.iter_num):
            for m, doc in enumerate(self.doc):
                for n, wd in enumerate(doc.words):
                    t = self.z[m][n]

                    wd_id = self.words.index(wd)
                    self.nw[wd_id][t] -= 1
                    self.nwsum[t] -= 1
                    self.nd[m][t] -= 1
                    self.ndsum[m] -= 1

                    p = [0 for _ in range(self.big_k)]
                    for k in range(self.big_k):
                        p[k] = (self.nw[wd_id][k]+self.beta)/(self.nwsum[k]+len(self.words)*beta)*(self.nd[m][k]+alpha)/(self.ndsum[k]+self.big_k*alpha)

                    new_t = self.cumulative(p)
                    self.z[m][n] = new_t
                    self.nw[wd_id][new_t] += 1
                    self.nwsum[new_t] += 1
                    self.nd[m][new_t] += 1
                    self.ndsum[m] += 1
                self._theta()
                self._phi()

    def cumulative(self, input_p):
        sum_p = np.sum(input_p)
        r = np.random.uniform(0, sum_p)

        sp = 0
        select_k = -1
        for i, p in enumerate(input_p):
            sp += p
            if r < sp:
                select_k = i
            else:
                break
        return select_k

    def _theta(self):
        for i in range(len(self.doc)):
            self.theta[i] = (self.nd[i]+alpha)/(self.ndsum[i]+self.big_k*alpha)

    def _phi(self):
        for i in range(self.big_k):
            self.phi[i] = (self.nw.T[i]+self.beta)/(self.nwsum[i]+len(self.words)*beta)

    def predict_one(self, one_doc_str):
        one_doc = list(jieba.cut(one_doc_str))
        doc_theta = np.zeros(self.big_k)
        doc_z = np.zeros(len(one_doc))
        doc_nd = np.zeros(self.big_k)
        doc_ndsum = 0

        for n, wd in enumerate(one_doc):
            topic_index = np.random.randint(0, self.big_k)
            doc_z[n] = topic_index
            # word_id = self.words.index(wd)

            doc_nd[topic_index] += 1
            doc_ndsum += 1

        for i in range(10):
            for n, wd in enumerate(one_doc):
                t = doc_z[n]

                doc_nd[t] -= 1
                p = [0 for _ in range(self.big_k)]
                for k in range(self.big_k):
                    p[k] = (doc_nd[k] + alpha) / (doc_ndsum + self.big_k * alpha)


if __name__ == "__main__":
    lda = LDA(doc_file, 5, 1, 1, 100)
    lda.sample()
