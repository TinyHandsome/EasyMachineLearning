#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=utf-8 

"""
@author: Liyingjun
@contact: 694317828@qq.com
@software: pycharm
@file: FigurePlot.py
@time: 2022/7/4 14:50
@desc: 绘图工具
        1. [plt字体和负号处理](https://blog.csdn.net/u010916338/article/details/96430916?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1)
        2. [plt.ticks字体大小设置参考](https://www.delftstack.com/zh/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/)
        3. [sns.heatmap参数详解1](https://blog.csdn.net/m0_38103546/article/details/79935671)
        4. [sns.heatmap参数详解2](https://www.pianshen.com/article/975899613/)
        5. [sns.heatmap刻度异常问题(旋转解决)](https://blog.csdn.net/chenhaouestc/article/details/79132602)
        6. [sns.pairplot参数详解](https://www.jianshu.com/|p/6e18d21a4cad)
        7. [sns.pairplot参数详解2](https://www.jianshu.com/p/c50cb4f1029f)
        8. [sns.pairplot控制图例位置](https://stackoverflow.com/questions/37815774/seaborn-pairplot-legend-how-to-control-position)
        9. [pl.legend图例参数详解](https://blog.csdn.net/helunqu2017/article/details/78641290?utm_source=blogxgwz6)
        10. [数据治理 | 随心所欲的Pandas绘图！](https://mp.weixin.qq.com/s/GZplBWrbkjMHsN7znWsRjA)
"""
import os.path

from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps

from model_structure.utils import generate_file_name

sns.set(style="ticks", color_codes=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FigurePlot:
    default_dpi = 500
    default_figsize=(10, 5)

    def savefig(self, name, save_path, prefix='', tight_layout=True):
        name = generate_file_name(name, prefix=prefix, suffix='.jpg')
        fig_path = os.path.join(save_path, name)

        if tight_layout:
            plt.savefig(fig_path, dpi=self.default_dpi, bbox_inches='tight')
            plt.tight_layout()
        else:
            plt.savefig(fig_path, dpi=self.default_dpi)

        return fig_path

    def plot_line(self, df: DataFrame, figure_save_path=None, style='-o', **kwargs):
        """
        绘制折线图
        :param df: df格式的数据，数据和label都有，横坐标是index
        """
        prefix = '折线图'

        # 自定义figsize
        ax = df.plot(kind='line', style=style, figsize=self.default_figsize, ms=3)

        if figure_save_path is not None:
            self.savefig('折线图', './', prefix=prefix)

