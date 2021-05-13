#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2021/2/20 20:52
    @Author  : jack.li
    @Site    : 
    @File    : evaluation.py
    增加评估ner 结果模型，使用严格策略，
    即位置和分类都对才能算抽取准确
"""


def extract_entity(input_label):
    start = None
    label = None
    extract_ner = []
    for i, x in enumerate(input_label):
        if x == "O":
            if start is not None:
                extract_ner.append((start, i, label))
                start = None
                label = None
        else:
            xindex, xlabel = x.split("-")
            if xindex == "B":
                if start is not None:
                    extract_ner.append((start, i, label))
                start = i
                label = xlabel
            else:
                if label != xlabel:
                    start = None
                    label = None
    return extract_ner


# 这里输入的BIO类型的数据
def metrix(true_labels, predict_labels):
    true_res = 0
    pred_res = 0
    predict_true = 0

    assert len(true_labels) == len(predict_labels)

    for i, label in enumerate(true_labels):
        pred_label = predict_labels[i]

        assert len(label) == len(pred_label)

        true_entity = extract_entity(label)
        pred_entity = extract_entity(pred_label)

        true_res += len(true_entity)
        pred_res += len(pred_entity)

        d_true = {(s, e): lb for s, e, lb in true_entity}
        d_pred = {(s, e): lb for s, e, lb in pred_entity}

        for k, v in d_true.items():
            if k in d_pred and d_pred[k] == v:
                predict_true += 1

    recall = predict_true*1.0/true_res
    precision = predict_true*1.0/pred_res

    return recall, precision


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
        "recall": (score + 1e-8) / (d_count + 1e-3),
        "precision": (score + 1e-8) / (p_count + 1e-3)
    }
    matrix["f1_value"] = 2 * matrix["recall"] * matrix["precision"] / (matrix["recall"] + matrix["precision"])
    return matrix


# 软评估结果, 不要求完全重合
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
