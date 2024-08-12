import numpy as np

from .. import norm


class CRITIC:
    def __init__(self, matrix: np.ndarray):
        """输入正向化指标矩阵"""
        self.matrix = matrix
        self.sigma: np.ndarray
        """对比强度"""
        self.R: np.ndarray
        """冲突指标"""
        self.C: np.ndarray
        """信息承载量"""
        self.W: np.ndarray
        """权重向量"""

    def normalize(self, norm_func=norm.min_max):
        """归一化化决策矩阵"""
        self.M = norm_func(self.matrix)

    def calc_comparative_strength(self):
        """计算对比强度"""
        self.sigma = np.std(self.matrix, axis=1)

    def calc_conflict_indicators(self):
        """计算冲突指标"""
        cor = np.corrcoef(self.matrix)
        self.R = np.sum(1 - cor, axis=1)

    def calc_information_carrying_capacity(self):
        """计算承载量"""
        self.C = self.sigma * self.R

    def calc_weight(self, norm_func=norm.prob):
        """计算权重"""
        self.W = norm_func(self.C)
        return self.W

    def run(self):
        self.normalize()
        self.calc_comparative_strength()
        self.calc_conflict_indicators()
        self.calc_information_carrying_capacity()
        self.calc_weight()
        return self.W

    def calc_score(self, data: np.ndarray, norm_func=lambda x: x) -> float:
        """计算得分"""
        data = np.array(data)
        data = norm_func(data)
        return self.W @ data
