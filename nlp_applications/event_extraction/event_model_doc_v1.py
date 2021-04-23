#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
"""
    文档级别的事件抽取模型
"""

import tensorflow as tf
from typing import List, Callable
import numpy as np
from nlp_applications.data_loader import LoaderBaiduDueeFin, EventDocument, Event, Argument

sample_path = "D:\\data\\篇章级事件抽取\\"
bd_data_loader = LoaderBaiduDueeFin(sample_path)

for doc in bd_data_loader.document:
    print(doc)
