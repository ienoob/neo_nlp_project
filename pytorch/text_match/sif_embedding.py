#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import jieba
import numpy as np
from sklearn.decomposition import TruncatedSVD



def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if  params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb

data_path = "D:\\data\\文本匹配\\paws-x-zh\\train.tsv"
dev_path = "D:\\data\\文本匹配\\paws-x-zh\\dev.tsv"

with open(data_path, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = f.read()

train_data_list = train_data.split("\n")
dev_data_list = dev_data.split("\n")

def load_word_vector(embedding_file):
    embedding_dict = dict()
    with open(embedding_file, encoding="utf-8") as f:
        for line in f:
            if len(line.rstrip().split(" ")) <= 2:
                continue
            token, vector = line.rstrip().split(" ", 1)
            embedding_dict[token] = np.fromstring(vector, dtype=np.float, sep=" ")
    return embedding_dict
def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
       raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

path = "D:\data\word2vec\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
        # f = open(path, "r", encoding="utf-8")
model = load_word_vector(path)

acc_num = 0.0
for data_item in train_data_list:
    data_item = data_item.strip()
    if len(data_item) == 0:
        continue
    if len(data_item.split("\t")) == 1:
        # print(data, "hello")
        continue

    sentence1, sentence2, label = data_item.split("\t")
    # sentence1_words = list(jieba.cut(sentence1))
    # sentence2_words = list(jieba.cut(sentence2))

    sentence1_words = [model[word] for word in jieba.cut(sentence1) if word in model]
    sentence2_words = [model[word] for word in jieba.cut(sentence2) if word in model]
    # s1_embed = None
    # for embed in sentence1_words:
    #     if s1_embed is None:
    #         s1_embed = embed
    #     else:
    #         s1_embed += embed
    #
    # s2_embed = None
    # for embed in sentence2_words:
    #     if s2_embed is None:
    #         s2_embed = embed
    #     else:
    #         s2_embed += embed

    sentence1 = np.array(sentence1_words)
    sentence2 = np.array(sentence2_words)

    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(sentence1)
    sentence1_sif = svd.components_

    svd.fit(sentence2)
    sentence2_sif = svd.components_

    score = cosine_distance(sentence1_sif, sentence2_sif)

    pred_value = 0
    if score > 0.5:
        pred_value = 1
    # score_list.append(score)
    if pred_value == int(label):
        acc_num += 1

print(acc_num/len(train_data_list))


