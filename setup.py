#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: js调用文件
"""

import sys
import json


arg_json = sys.argv[1]
arg_dict = json.loads(arg_json)


X = arg_dict.get('X')
y = arg_dict.get('y')

print(X, y)