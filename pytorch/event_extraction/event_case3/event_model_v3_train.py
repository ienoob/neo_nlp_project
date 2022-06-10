#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
from pytorch.event_extraction.event_model_v3 import train

from pytorch.event_extraction.event_case3.train_data import rt_data


if __name__ == "__main__":
    train(rt_data)
