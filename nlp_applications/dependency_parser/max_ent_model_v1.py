#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/7/11 17:22
    @Author  : jack.li
    @Site    : 
    @File    : max_ent_model_v1.py

"""
import logging

import numpy as np

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaxEntropyModel(object):

    def __init__(self, input_x, input_y):

        self.input_x = input_x
        self.input_y = input_y

        self.label = {label for label in input_y}
        self.label_num = len(self.label)
        self.data_num = len(self.input_x)
        # feature_num = {(r, y) for row in input_x for r in row for y in input_y}
        self.p_hat_xy = dict()
        self.p_hat_x = dict()
        for i, row in enumerate(input_x):
            yi = input_y[i]
            for j, r in enumerate(row):
                ky = ("{}-{}".format(j, r), yi)
                self.p_hat_xy.setdefault(ky, 0.0)
                self.p_hat_xy[ky] += 1.0/self.data_num

                self.p_hat_x.setdefault("{}-{}".format(i, r), 0)
                self.p_hat_x["{}-{}".format(i, r)] += 1.0/self.data_num
            # print(p_hat_xy)

        self.feature_func2id = dict()
        for k, _ in self.p_hat_xy.items():
            self.feature_func2id[k] = len(self.feature_func2id)
        self.id2feature = {v: k for k, v in self.feature_func2id.items()}
        # for k, v in p_hat_xy.items():
        #     logger.info((k, v))

        self.feature_num = len(self.p_hat_xy)
        logger.info("feature num {}".format(self.feature_num))

        self.w = [0.0] * self.feature_num
        self.column_num = len(input_x[0])
        self.big_m = 1000
        # for k, v in p_hat_x.items():
        #     logger.info()

    def calc_pw_xy(self, x_row):
        big_z_value = [0.0] * self.label_num
        for yi in range(self.label_num):
            for j in range(self.column_num):
                r = x_row[j]
                key = ("{}-{}".format(j, r), yi)
                indx = self.feature_func2id[key]
                big_z_value[yi] += self.w[indx]
        biz_z_value = np.exp(big_z_value)
        biz_z_value_p = biz_z_value/np.sum(big_z_value)

        return biz_z_value_p


    def calculate_hp_xy(self):
        hp_exy = [0.0] * self.feature_num
        for i in range(self.data_num):
            biz_z_value_p = self.calc_pw_xy(self.input_x[i])

            hp_exy[i] = np.sum([np.log(p)*(-p) for p in biz_z_value_p])

        return hp_exy

    def train(self):
        for i in range(10):
            hp_exy = self.calculate_hp_xy()

            for j in range(self.feature_num):
                key = self.id2feature[j]
                e_p_hat = self.p_hat_xy[key]
                e_p = hp_exy[j]

                self.w = [wi+(1.0/self.big_m)*np.log(e_p_hat/e_p) for wi in self.w]
            print(np.sum(hp_exy))





def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    # 存放数据及标记的list
    dataList = []; labelList = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')
        #二分类，list中放置标签
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)
        #二值化
        dataList.append([int(int(num) > 128) for num in curLine[1:]])

    #返回data和label
    return dataList, labelList

if __name__ == "__main__":

    trainData, trainLabel = loadData('D:\data\\transMnist\Mnist\\mnist_train.csv')

    maxEnt = MaxEntropyModel(trainData[:20000], trainLabel[:20000])
