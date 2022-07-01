#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: js调用文件
"""

import sys
import json

from model_structure.ClassifierModels import get_classifier_info
from model_structure.ClassifierModels import MyRF

arg_json = sys.argv[1]
arg_dict: dict = json.loads(arg_json)

flag = arg_dict.get('flag')

# 获取分类器信息
classifier_classes_dict = get_classifier_info()

if flag == 'get_classifier_info':
    print(classifier_classes_dict.keys())


def test1():
    X = arg_dict.get('X')
    y = arg_dict.get('y')

    clf = MyRF()
    print(clf.english_name)
    result = clf.simple_model(X, y, model_save_path='./')
    print(result)
