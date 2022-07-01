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
from model_structure.RegressorModels import get_regressor_info
from model_structure.ClassifierModels import MyRF

arg_json = sys.argv[1]
arg_dict: dict = json.loads(arg_json)

method_name = arg_dict.get('flag')

# 获取分类器信息
classifier_classes_dict = get_classifier_info()
# 获取回归器信息
regressor_classes_dict = get_regressor_info()

if method_name == 'get_classifier_info':
    """获取分类信息"""
    print(classifier_classes_dict.keys())

if method_name == 'get_regressor_info':
    """获取回归信息"""
    print(regressor_classes_dict.keys())


if method_name == 'simple_model':
    """建立简单模型"""
    model_names: list = arg_dict.get('model_names')
    result_dict = {}
    for model_name in model_names:
        ...



def test1():
    X = arg_dict.get('X')
    y = arg_dict.get('y')

    clf = MyRF()
    print(clf.english_name)
    result = clf.simple_model(X, y, model_save_path='./')
    print(result)
