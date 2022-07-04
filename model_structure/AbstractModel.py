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

import os
import joblib
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, \
    mean_absolute_error

# 禁用警告显示
# import warnings
# warnings.filterwarnings('ignore')

from model_structure.utils import generate_file_name


class Model(metaclass=ABCMeta):
    """
        基本上所有模型的抽象类，包括分类和回归
    """
    random_state = 42

    # 模型名称
    @property
    @abstractmethod
    def name(self):
        ...

    # 模型类型
    @property
    @abstractmethod
    def model_type(self):
        ...

    # 模型中文名称
    @property
    @abstractmethod
    def chinese_name(self):
        ...

    # 模型英文名称
    @property
    @abstractmethod
    def english_name(self):
        ...

    # 模型参数
    @property
    @abstractmethod
    def parameters(self):
        ...

    # 模型默认评价指标
    @property
    @abstractmethod
    def default_scoring(self):
        ...

    # 模型的类
    @property
    @abstractmethod
    def model(self):
        ...

    # 支持的metrics
    @property
    @abstractmethod
    def model_metrics(self):
        ...

    def modeling(self, X, y, params=None, model_save_path=None, model_save_name=None):
        """【建模】
        :return: 返回训练的模型
        """
        # 不是所有的模型都可以设置随机种子
        try:
            model = self.model(random_state=self.random_state)
        except Exception as e:
            model = self.model()

        if params is not None:
            model.set_params(**params)
        model.fit(X, y)

        self.save_model_by_joblib(model, model_save_path, model_save_name)

        return model

    def save_model_by_joblib(self, model, model_save_path, model_save_name):
        """保存模型"""
        if model_save_path is not None:
            with open(os.path.join(model_save_path, model_save_name), 'wb') as f:
                joblib.dump(model, f)

    def simple_model(self, X, y, test_size=0.33, scoring: None or list = None, params=None, model_save_path=None,
                     **kwargs):
        """
        用默认参数进行建模，划分数据集
        :params: 是否指定参数
        :model_save_path: 是否保存模型，默认保存，保存的路径为当前路径
        :return: 评价指标的字典
        """
        prefix = '简单建模'

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state)

        model = self.modeling(X_train, y_train, params)
        y_pred = model.predict(X_test)

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

        # 生成当前的模型
        self.modeling(X, y, params, model_save_path, generate_file_name(self.name, prefix))

        return dict(result)

    def cv_model(self, X, y, cv=5, scoring: None or list = None, params=None, model_save_path=None, **kwargs):
        """用默认参数进行建模，用交叉验证进行验证"""
        prefix = '交叉验证'

        model = self.modeling(X, y, params)
        if scoring is None:
            scoring = [self.default_scoring]

        result = []
        for s in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=s)
            mean_score = np.mean(scores)
            result.append((s, mean_score))

        # 生成当前的模型
        self.modeling(X, y, params, model_save_path, generate_file_name(self.name, prefix))

        return dict(result)

    def param_search_model(self, X, y, cv=5, scoring: None or list = None, params=None, model_save_path=None, **kwargs):
        """用【网格搜索】交叉验证的方式进行建模"""
        prefix = '网格搜索'

        model = self.modeling(X, y, params)
        if scoring is None:
            scoring = [self.default_scoring]

        # 获取调参信息
        parameters_dict = kwargs.get('parameters_dict')
        if parameters_dict is None:
            parameters_dict = self.parameters

        result = []
        for s in scoring:
            grid = GridSearchCV(model, parameters_dict, cv=cv, scoring=s, return_train_score=True)
            grid.fit(X, y)

            # 保存调参信息
            # cv_results = pd.DataFrame(grid.cv_results_)
            # cv_results.to_excel(save_path_name, index_label='编号')

            result.append((s, {'best_params': grid.best_params_, 'best_score': grid.best_score_}))

            pp_prefix = '(scoring:' + s + ')' + prefix

            # 生成不同评价指标的最优参数模型
            self.save_model_by_joblib(grid.best_estimator_, model_save_path, generate_file_name(self.name, pp_prefix))

        return dict(result)


class MyClassifier(Model):
    """分类模型"""
    default_scoring = 'accuracy'
    model_type = 'classifier'
    model_metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }


class MyRegressor(Model):
    """回归模型"""
    default_scoring = 'r2'
    model_type = 'regressor'
    model_metrics = {
        'neg_mean_squared_error': mean_squared_error,
        'neg_mean_absolute_error': mean_absolute_error,
        'r2': r2_score
    }


if __name__ == '__main__':
    print(MyClassifier.__class__.__bases__[0].__dict__)
