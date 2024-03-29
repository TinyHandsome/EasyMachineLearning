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
def generate_file_name(name, prefix='', suffix='model'):
    """【保存】生成模型的名称"""
    current_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    result_name = '[' + current_time + '] ' + prefix + '_' + name + '.' + suffix
    return result_name


def dict_to_json_and_print(result):
    """将字典转为json的，打印输出"""
    json_result = json.dumps(result)
    print(json_result)


def get_json_attribute(json_dict, attr):
    """获取json数据中的对应的值"""
    my_dict = json.loads(json_dict)
    return my_dict.get(attr)


# 模型处理
def predict_from_model(model_path, X):
    """【选择模型的路径，进行预测】默认joblib
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    pred = model.predict(X)
    return {'y_pred': pred}


if __name__ == '__main__':
    generate_file_name('a')
