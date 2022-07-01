#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc: 辅助工具
        1. 模型名生成函数

"""
import json
import time
import joblib

# 常见处理
from model_structure.ClassifierModels import get_classifier_class
from model_structure.RegressorModels import get_regressor_class


def generate_model_name(model_name, prefix=''):
    """【保存】生成模型的名称"""
    current_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    name = '[' + current_time + '] ' + prefix + '_' + model_name + '.model'
    return name


def dict_to_json_and_print(result):
    """将字典转为json的，打印输出"""
    json_result = json.dumps(result)
    print(json_result)


# 模型处理
def get_X_y_modelSavePath_modelType_from_args(arg_dict):
    """获取 简单建模、交叉验证 需要的参数"""
    X = arg_dict.get('X')
    y = arg_dict.get('y')
    model_save_path = arg_dict.get('model_save_path')
    model_type = arg_dict.get('model_type')
    model_names: list = arg_dict.get('model_names')

    model_classes = None
    if model_type == 'classifier':
        model_classes = get_classifier_class()
    elif model_type == 'regressor':
        model_classes = get_regressor_class()

    return X, y, model_save_path, model_classes, model_names


def predict_from_model(model_path, X):
    """【选择模型的路径，进行预测】默认joblib
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    pred = model.predict(X)
    return {'y_pred': pred}
