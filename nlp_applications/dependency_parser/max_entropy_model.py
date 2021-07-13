#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import time
import numpy as np


class MaxEntropyModel(object):

    def __init__(self, input_x_list, input_y_list):
        assert len(input_x_list) == len(input_y_list)
        self.data_num = len(input_x_list)
        self.feature_num = len(input_x_list[0])
        self.class_value = list({y for y in input_y_list})
        self.class_num = len(self.class_value)
        self.train_x = input_x_list

        self.feature_func_dict = dict()
        self.feature_x_dict = dict()
        self.f_func2id = dict()

        for ri, data_row in enumerate(input_x_list):
            for fi in range(self.feature_num):
                f_key = "{0}-{1}-{2}".format(fi, data_row[fi], input_y_list[ri])
                self.feature_func_dict.setdefault(f_key, 0)
                self.feature_func_dict[f_key] += 1.0/self.data_num

                if f_key not in self.f_func2id:
                    self.f_func2id[f_key] = len(self.f_func2id)

                fx_key = "{0}-{1}".format(fi, data_row[fi])
                self.feature_x_dict.setdefault(fx_key, 0)
                self.feature_x_dict[fx_key] += 1.0/self.data_num

        self.f_func_num = len(self.feature_func_dict)
        self.id2f_func = {v:k for k, v in self.f_func2id.items()}
        self.ep_hat = self.get_ep_hat()
        self.py_x = None
        self.w = np.zeros(self.f_func_num)
        self.big_m = 10000

    def get_ep_hat(self):
        ep_hat = np.zeros(self.f_func_num)
        for iv in range(self.f_func_num):
            ep_hat[iv] = self.feature_func_dict[self.id2f_func[iv]]
        return ep_hat

    def get_ep(self):

        ep = np.ones(self.f_func_num)*1e-10

        for row_x in self.train_x:
            py_x = self.calc_py_x_row(row_x)
            # print(py_x, "py_x")

            for iv in range(self.feature_num):
                x_key = "{}-{}".format(iv, row_x[iv])
                for y_indx, y_value in enumerate(self.class_value):
                    # print(self.feature_x_dict[x_key])
                    f_key = "{0}-{1}-{2}".format(iv, row_x[iv], y_value)
                    if f_key in self.f_func2id:
                        f_id = self.f_func2id[f_key]
                        ep[f_id] += self.feature_x_dict[x_key] * py_x[y_indx] / self.data_num

        return ep

    def calc_py_x(self):
        self.py_x = np.zeros(self.f_func_num)

        for iv in range(self.f_func_num):
            func_key = self.id2f_func[iv]
            # y_value = int(func_key.split("-")[2])
            x_key = "-".join(func_key.split("-")[:2])

            self.py_x[iv] = np.exp(self.w[iv] * self.feature_func_dict[func_key])
            big_z = 0.0
            for y in self.class_value:
                _func_key = "{}-{}".format(x_key, y)
                if _func_key in self.f_func2id:

                    _func_key_id = self.f_func2id[_func_key]
                    big_z += np.exp(self.w[_func_key_id] * self.feature_func_dict[_func_key])
                else:
                    big_z += np.exp(0)
            self.py_x[iv] /= big_z

    def calc_py_x_row(self, x_row):
        py_x = np.zeros(self.class_num)
        for yi, y in enumerate(self.class_value):
            for fi, xi in enumerate(x_row):
                f_key = "{0}-{1}-{2}".format(fi, xi, y)
                if f_key in self.f_func2id:
                    f_ind = self.f_func2id[f_key]
                    py_x[yi] += self.w[f_ind]
        py_x = np.exp(py_x)
        py_x /= np.sum(py_x)

        return py_x

    def train(self, iter_num):
        for _ in range(iter_num):
            ep = self.get_ep()
            print(np.mean(ep))

            for iv in range(self.f_func_num):
                self.w[iv] += (1.0/self.big_m)*np.log(self.ep_hat[iv]/ep[iv])

    def predict(self, input_x_list):
        self.calc_py_x()
        predict_list = []
        for x_row in input_x_list:
            pred_value = self.calc_py_x_row(x_row)
            predict_list.append(np.argmax(pred_value))
        return predict_list

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


if __name__ == '__main__':
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainData, trainLabel = loadData('D:\data\\transMnist\Mnist\\mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testData, testLabel = loadData('D:\data\\transMnist\Mnist\\mnist_test.csv')

    max_ent = MaxEntropyModel(trainData[:20000], trainLabel[:20000])
    max_ent.train(10)

    predict_res = max_ent.predict(testData)

    accuracy = 0.0
    for iv, value in enumerate(testLabel):
        if predict_res[iv] == value:
            accuracy += 1

    print(accuracy/len(testLabel))





