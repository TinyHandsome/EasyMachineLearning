#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@contact: litian_cup@163.com
@desc: 模型的抽象类
        参考：
            1. [回归问题评价指标](https://wenku.baidu.com/view/c442b4636aeae009581b6bd97f1922791688be22.html)
"""

from tkinter.tix import DirTree
import pandas as pd
from abc import ABCMeta, abstractclassmethod, abstractproperty
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error


class Model(metaclass=ABCMeta):
    """
        基本上所有模型的抽象类，包括分类和回归
    """
    random_state = 42

    # 模型名称
    @abstractproperty
    def name(self): ...

    # 模型类型
    @abstractproperty
    def model_type(self): ...

    # 模型中文名称
    @abstractproperty
    def chinese_name(self): ...

    # 模型英文名称
    @abstractproperty
    def english_name(self): ...

    # 模型参数
    @abstractproperty
    def parameters(self): ...

    # 模型默认评价指标
    @abstractproperty
    def default_scoring(self): ...

    # 模型的类
    @abstractproperty
    def model(self): ...

    # 支持的metrics
    @abstractproperty
    def model_metrics(self): ...

    def simple_model(self, X, y, test_size=0.33, scoring: None or list = None, params=None):
        """
        用默认参数进行建模，划分数据集
        :params: 是否指定参数
        :return: 评价指标的字典
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state)

        clf = self.model(random_state=self.random_state)
        if params is not None:
            clf.set_params(**params)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # true_pred = pd.DataFrame({'测试编号': range(len(y_test)), '真实值': y_test, '预测值': y_pred})
        # true_pred.set_index('测试编号', inplace=True)

        if scoring is None:
            scoring_func = [
                (self.default_scoring, self.model_metrics.get(self.default_scoring))]
        else:
            scoring_func = [(s, self.model_metrics.get(s)) for s in scoring]

        result = []
        for s, s_func in scoring_func:
            result.append((s, s_func(y_test, y_pred)))

        return dict(result)


class MyClassifier(Model):
    """分类模型"""
    default_scoring = 'accuracy'
    model_type = '分类'
    model_metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }


class MyRegressor(Model):
    """回归模型"""
    default_scoring = 'r2'
    model_type = '回归'
    model_metrics = {
        'neg_mean_squared_error': mean_squared_error,
        'neg_mean_absolute_error': mean_absolute_error,
        'r2': r2_score
    }
