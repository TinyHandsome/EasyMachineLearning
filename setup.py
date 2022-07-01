#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: js调用文件
"""

import sys
import json

from model_structure.AbstractModel import Model
from model_structure.ClassifierModels import get_classifier_info
from model_structure.RegressorModels import get_regressor_info

arg_json = sys.argv[1]
arg_dict: dict = json.loads(arg_json)

method_name = arg_dict.get('method_name')

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

    # 需要出入额外的参数
    X = arg_dict.get('X')
    y = arg_dict.get('y')
    model_save_path = arg_dict.get('model_save_path')

    model_names: list = arg_dict.get('model_names')
    result_dict = {}
    for model_name in model_names:
        model: Model = classifier_classes_dict.get(model_name)
        result_dict[model_name] = model.simple_model(X, y, model_save_path=model_save_path)

    print(result_dict)
