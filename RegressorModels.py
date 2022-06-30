#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc:
"""

from AbstractModel import MyRegressor
from sklearn.datasets import load_boston

from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor


# 1. Linear Regression / 线性回归
class MyLR(MyRegressor):
    name = "线性回归"
    chinese_name = "线性回归"
    english_name = "Linear Regression"
    parameters = None
    model = LinearRegression


# 2. Decision Tree Regressor / 决策树回归
class MyDT(MyRegressor):
    name = "决策树"
    chinese_name = "决策树回归"
    english_name = "DecisionTree Regressor"
    parameters = OrderedDict([
        # 正则化一般选择l2就ok了
        # 参数solver的选择，如果是L2正则化，那么4种可选的算法{'newton-cg', 'lbfgs', 'liblinear', 'sag'}都可以选择
        # 但是如果penalty是L1正则化的话，就只能选择'liblinear'了
        ('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag']),
        # 惩罚因子的调参与支持向量机一致
        ('C', [0.01, 0.1, 1, 10, 100])
    ])
    model = DecisionTreeRegressor


# 3. 支持向量机回归
class MySVR(MyRegressor):
    name = "支持向量机回归"
    chinese_name = "支持向量机"
    english_name = "Support Vector Machine"
    parameters = OrderedDict([
        # C表示模型对误差的惩罚系数，C越大，模型越容易过拟合；C越小，模型越容易欠拟合
        ('C', [0.1, 1, 10]),
        # gamma反映了数据映射到高维特征空间后的分布，gamma越大，支持向量越多，gamma值越小，支持向量越少
        # gamma越小，模型的泛化性变好，但过小，模型实际上会退化为线性模型；gamma越大，理论上SVM可以拟合任何非线性数据
        ('gamma', [1, 0.1, 0.01])
    ])
    model = SVR


# 4. KNN Regressor / K最近邻回归
class MyKNN(MyRegressor):
    name = "KNN"
    chinese_name = "K最近邻"
    english_name = "KNN Regressor"
    parameters = OrderedDict([
        # n_neighbors: int, 可选参数(默认为 5)，每个点与周围多少个点进行比较
        ('n_neighbors', range(1, 10, 1))
    ])
    model = KNeighborsRegressor


# 5. Random Forest Regressor / 随机森林回归
class MyRF(MyRegressor):
    name = "RandomForestRegressor"
    chinese_name = "随机森林回归"
    english_name = "RandomForest Regressor"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = RandomForestRegressor


# 6. Adaboost Regressor / 自适应提升法
class MyAdaboost(MyRegressor):
    name = "Adaboost"
    chinese_name = "自适应提升算法"
    english_name = "Adaptive Boosting"

    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('base_estimator__max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('base_estimator__min_samples_split', list(range(2, 10, 1))[::-1])
    ])

    model = AdaBoostRegressor


# 7. GBDT(Gradient Boosting Decision Tree) Regressor / 梯度提升决策树
class MyGBDT(MyRegressor):
    name = "GBDT"
    chinese_name = "梯度提升决策树"
    english_name = "Gradient Boosting Decision Tree Regressor"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = GradientBoostingRegressor


# 8. Bagging Regressor / Bagging回归
class MyBR(MyRegressor):
    name = "Bagging回归"
    chinese_name = "Bagging回归"
    english_name = "Bagging Regressor"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('base_estimator__max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('base_estimator__min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = BaggingRegressor


# 9. ExtraTreesRegressor / 极端随机树回归
class MyETR(MyRegressor):
    name = "ExtraTrees"
    chinese_name = "极端梯度提升树"
    english_name = "ExtraTrees Regressor"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = ExtraTreesRegressor


# 10. XGBoost / 极端梯度提升
class MyXGBoost(MyRegressor):
    name = "XGBoost"
    chinese_name = "极端梯度提升树"
    english_name = "XGBoost"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = XGBRegressor


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    reg = MyBR()
    result = reg.simple_model(X, y)
    print(result)













