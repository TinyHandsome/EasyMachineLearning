#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8

"""
@author: Li Tian
@desc:
        [正则化的理解](https://www.zhihu.com/question/20924039)
        [惩罚参数C的理解](https://blog.csdn.net/csdn_lzw/article/details/80185529)
        我的理解：
            惩罚参数C越大 --> 越不允许出错 --> 偏向过拟合 --> 需要求解的系数越多 --> 正则化系数lamda越小

        [多项式朴素贝叶斯参数详解](https://www.jianshu.com/p/17b04a5b6410)
        [朴素贝叶斯三种方法的使用区别](https://blog.csdn.net/brucewong0516/article/details/78798359)
        [KNN参数详解](https://www.jianshu.com/p/871884bb4a75)
        [LR参数详解](https://www.jianshu.com/p/99ceb640efc5)
        [LR调参](https://www.cnblogs.com/webRobot/p/11781078.html)
        [SVM参数详解](https://blog.csdn.net/sun_shengyun/article/details/55669160)
        [SVM调参](https://www.cnblogs.com/yuehouse/p/9742697.html)
        [决策树参数详解](https://www.cnblogs.com/lyxML/p/9575820.html)
        [AdaBoost参数详解](https://www.cnblogs.com/mdevelopment/p/9445090.html)

"""
from model_structure.AbstractModel import MyClassifier

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from collections import OrderedDict


# 1. Multinomial Naive Bayes Classifier / 多项式朴素贝叶斯
class MyMNB(MyClassifier):
    name = "MNB"
    chinese_name = "多项式朴素贝叶斯"
    english_name = "Multinomial Naive Bayes Classifier"
    parameters = OrderedDict([
        # alpha：拉普拉修/Lidstone平滑参数,浮点型,可选项,默认1.0
        ('alpha', [0, 0.25, 0.5, 0.75, 1])
    ])
    model = MultinomialNB


# 2. Gaussian Naive Bayes Classifier / 高斯朴素贝叶斯
class MyGNB(MyClassifier):
    name = "GNB"
    chinese_name = "高斯朴素贝叶斯"
    english_name = "Gaussian Naive Bayes Classifier"
    # 几乎没有参数可调
    parameters = None
    model = GaussianNB


# 3. KNN Classifier / K最近邻
class MyKNN(MyClassifier):
    name = "KNN"
    chinese_name = "K最近邻"
    english_name = "KNN Classifier"
    parameters = OrderedDict([
        # n_neighbors: int, 可选参数(默认为 5)，每个点与周围多少个点进行比较
        ('n_neighbors', range(1, 10, 1))
    ])
    model = KNeighborsClassifier


# 4. Logistic Regression Classifier / 逻辑回归
class MyLR(MyClassifier):
    name = "LR"
    chinese_name = "逻辑回归"
    english_name = "Logistic Regression Classifier"
    parameters = OrderedDict([
        # 正则化一般选择l2就ok了
        # 参数solver的选择，如果是L2正则化，那么4种可选的算法{'newton-cg', 'lbfgs', 'liblinear', 'sag'}都可以选择
        # 但是如果penalty是L1正则化的话，就只能选择'liblinear'了
        ('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag']),
        # 惩罚因子的调参与支持向量机一致
        ('C', [0.01, 0.1, 1, 10, 100])
    ])
    model = LogisticRegression


# 5. SVM Classifier / 支持向量机
class MySVM(MyClassifier):
    name = "SVC"
    chinese_name = "支持向量机"
    english_name = "Support Vector Machine"
    parameters = OrderedDict([
        # C表示模型对误差的惩罚系数，C越大，模型越容易过拟合；C越小，模型越容易欠拟合
        ('C', [0.1, 1, 10]),
        # gamma反映了数据映射到高维特征空间后的分布，gamma越大，支持向量越多，gamma值越小，支持向量越少
        # gamma越小，模型的泛化性变好，但过小，模型实际上会退化为线性模型；gamma越大，理论上SVM可以拟合任何非线性数据
        ('gamma', [1, 0.1, 0.01])
    ])
    model = SVC


# 6. Decision Tree Classifier / 决策树
class MyDT(MyClassifier):
    name = "DT"
    chinese_name = "决策树"
    english_name = "Decision Tree Classifier"
    parameters = OrderedDict([
        # 特征选择标准，可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益
        ('criterion', ['gini', 'entropy']),
        # 最大树深度越小越简单
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = DecisionTreeClassifier


# 7. Random Forest Classifier / 随机森林
class MyRF(MyClassifier):
    name = "RF"
    chinese_name = "随机森林"
    english_name = "Random Forest Classifier"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = RandomForestClassifier


# 8. GBDT(Gradient Boosting Decision Tree) Classifier / 梯度提升决策树
class MyGBDT(MyClassifier):
    name = "GBDT"
    chinese_name = "梯度提升决策树"
    english_name = "Gradient Boosting Decision Tree Classifier"
    parameters = OrderedDict([
        # 集成模型数量越小越简单
        ('n_estimators', range(10, 500, 20)),
        # 最大树深度越小越简单，集成学习的树不能太深
        ('max_depth', range(3, 10, 1)),
        # 最小样本分割数越大越简单
        ('min_samples_split', list(range(2, 10, 1))[::-1])
    ])
    model = GradientBoostingClassifier


# 9. XGBoost / 极端梯度提升
class MyXGBoost(MyClassifier):
    name = "XGB"
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
    model = XGBClassifier


# 10. AdaBoost Classifier / 自适应提升法
class MyAdaboost(MyClassifier):
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

    model = AdaBoostClassifier


def get_classifier_info() -> dict:
    """
    获取当前分类器的介绍信息和类
    """
    current_classifier_classes_dict = {
        'chinese': {
            MyMNB.name: MyMNB.chinese_name,
            MyGNB.name: MyGNB.chinese_name,
            MyKNN.name: MyKNN.chinese_name,
            MyLR.name: MyLR.chinese_name,
            MySVM.name: MySVM.chinese_name,
            MyDT.name: MyDT.chinese_name,
            MyRF.name: MyRF.chinese_name,
            MyGBDT.name: MyGBDT.chinese_name,
            MyXGBoost.name: MyXGBoost.chinese_name,
            MyAdaboost.name: MyAdaboost.chinese_name
        },
        'english': {
            MyMNB.name: MyMNB.english_name,
            MyGNB.name: MyGNB.english_name,
            MyKNN.name: MyKNN.english_name,
            MyLR.name: MyLR.english_name,
            MySVM.name: MySVM.english_name,
            MyDT.name: MyDT.english_name,
            MyRF.name: MyRF.english_name,
            MyGBDT.name: MyGBDT.english_name,
            MyXGBoost.name: MyXGBoost.english_name,
            MyAdaboost.name: MyAdaboost.english_name
        }
    }

    return current_classifier_classes_dict


def get_classifier_class() -> dict:
    """
    获取当前分类器的介绍信息和类
    """
    current_classifier_classes_dict = {
        MyMNB.name: MyMNB,
        MyGNB.name: MyGNB,
        MyKNN.name: MyKNN,
        MyLR.name: MyLR,
        MySVM.name: MySVM,
        MyDT.name: MyDT,
        MyRF.name: MyRF,
        MyGBDT.name: MyGBDT,
        MyXGBoost.name: MyXGBoost,
        MyAdaboost.name: MyAdaboost
    }

    return current_classifier_classes_dict


if __name__ == '__main__':
    print(get_classifier_info())
