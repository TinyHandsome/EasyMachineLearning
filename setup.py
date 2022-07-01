#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: js调用文件
"""
import json
import sys

from model_structure.AbstractModel import Model
from model_structure.ClassifierModels import get_classifier_info, get_classifier_class
from model_structure.RegressorModels import get_regressor_info, get_regressor_class
from model_structure.utils import dict_to_json_and_print

arg_json = sys.argv[1]
arg_dict: dict = json.loads(arg_json)

method_name = arg_dict.get('method_name')


if method_name == 'get_classifier_info':
    """获取分类信息"""
    classifier_classes_info = get_classifier_info()
    dict_to_json_and_print(classifier_classes_info)

if method_name == 'get_regressor_info':
    """获取回归信息"""
    regressor_classes_info = get_regressor_info()
    dict_to_json_and_print(regressor_classes_info)


def get_X_y_modelSavePath_modelType_from_args(args):
    """获取 简单建模、交叉验证 需要的参数"""
    X = args.get('X')
    y = args.get('y')
    model_save_path = args.get('model_save_path')
    model_type = args.get('model_type')
    model_names: list = args.get('model_names')

    model_classes = None
    if model_type == 'classifier':
        model_classes = get_classifier_class()
    elif model_type == 'regressor':
        model_classes = get_regressor_class()

    return X, y, model_save_path, model_classes, model_names


if method_name == 'simple_model':
    """建立简单模型"""

    # 需要出入额外的参数
    X, y, model_save_path, model_classes, model_names = get_X_y_modelSavePath_modelType_from_args(arg_dict)

    result_dict = {}
    for model_name in model_names:
        model: Model = model_classes.get(model_name)
        result_dict[model_name] = model.simple_model(X, y, model_save_path=model_save_path)

    dict_to_json_and_print(result_dict)

if method_name == 'cv_model':
    """建立交叉验证评估的模型"""

    # 需要出入额外的参数
    X, y, model_save_path, model_classes, model_names = get_X_y_modelSavePath_modelType_from_args(arg_dict)

    result_dict = {}
    for model_name in model_names:
        model: Model = model_classes.get(model_name)
        result_dict[model_name] = model.cv_model(X, y, model_save_path=model_save_path)

    dict_to_json_and_print(result_dict)
