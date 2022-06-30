#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: js调用文件
"""

import sys
import json

from ClassifierModels import MyRF


arg_json = sys.argv[1]
arg_dict = json.loads(arg_json)


X = arg_dict.get('X')
y = arg_dict.get('y')

clf = MyRF()
print(clf.english_name)
result = clf.simple_model(X, y)
print(result)