#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***


def eval_metrix(hit_num, true_num, predict_num):
    recall = (hit_num + 1e-8) / (true_num + 1e-3)
    precision = (hit_num + 1e-8) / (predict_num + 1e-3)
    f1_value = 2 * recall * precision / (recall + precision)

    return {
        "recall": recall,
        "precision": precision,
        "f1_value": f1_value
    }