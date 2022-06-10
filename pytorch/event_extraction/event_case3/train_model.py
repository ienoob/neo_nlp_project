#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from pytorch.event_extraction.event_model_v1 import train_model

from pytorch.event_extraction.event_case3.train_data import rt_data


if __name__ == "__main__":
    train_model(rt_data, "bidding", "all")
