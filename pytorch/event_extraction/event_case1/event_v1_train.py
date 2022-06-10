#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from pytorch.event_extraction.event_model_v1 import train_model, save_load_model
from pytorch.event_extraction.event_case1.train_data_v2 import rt_data


if __name__ == "__main__":
    train_model(rt_data, "finance", "all")
    # save_load_model(rt_data)
