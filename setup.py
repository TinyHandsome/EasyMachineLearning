#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: js调用文件
        1. [python根据字符串调用函数](https://www.jianshu.com/p/a7b69a4e9ed7)
"""
import json
import sys
import numpy as np

from model_structure.ClassifierModels import get_classifier_info, get_classifier_class
from model_structure.RegressorModels import get_regressor_info, get_regressor_class
from model_structure.utils import dict_to_json_and_print

# 测试部分 TODO
# 分类问题测试
# from sklearn.datasets import load_iris
# method_name = 'simple_model'
# X, y = load_iris(return_X_y=True, as_frame=True)
# arg_dict = {
#     "X": X,
#     "y": y,
#     "model_save_path": "./",
#     "model_type": "classifier",
#     "model_names": ["MNB", "GNB", "KNN", "LR", "SVC", "DT", "RF", "GBDT", "XGB", "Adaboost"]
# }
# 回归问题测试
# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)
# method_name = 'simple_model'
# arg_dict = {
#     "X": X,
#     "y": y,
#     "model_save_path": "./",
#     "model_type": "regressor",
#     "model_names": ["LR", "DT", "SVR", "KNN", "RF", "Adaboost", "GBDT", "BR", "ETR", "XGB"]
# }

# 参数获取部分，测试要注释这部分代码
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


# 建模集合
if method_name in ['simple_model', 'cv_model', 'param_search_model']:
    """建立
        【简单模型】
        【交叉验证评估的模型】
        【建立网格搜索的模型，也可能是其他搜索，以后再补充】
    """
    # 需要出入额外的参数
    X, y, model_save_path, model_classes, model_names = get_X_y_modelSavePath_modelType_from_args(arg_dict)
    # 处理X和y为Array类型的数据
    X = np.array(X)
    y = np.array(y)

    result_dict = {}
    for model_name in model_names:
        MyModel = model_classes.get(model_name)
        model = MyModel()
        method_model = getattr(model, method_name)
        result_dict[model_name] = method_model(X=X, y=y, model_save_path=model_save_path)

    dict_to_json_and_print(result_dict)


