#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***

# token pair 的方式
# 1) hidden1 concat hidden2 tplink
# 2) hidden1 + hidden2 tplink
# 3) conditional layer normalization tplink
# 4) attention, qvk   global pointer
# 5) biaffine
# 6) multihead select
