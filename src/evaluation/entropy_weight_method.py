import numpy as np

from ..norm import min_max, prob


class EWM:
    """
    熵权法（Entropy Weight Method, EWM）是一种客观赋权方法，基于信息论的理论基础，
    根据各指标的数据的分散程度，利用信息熵计算并修正得到各指标的熵权，较为客观。
    相对而言这种数据驱动的方法就回避了上面主观性因素造成的重复修正的影响。
    """

    def __init__(self, matrix: np.ndarray):
        """
        Parameters
        ---
        matrix: np.ndarray
            已经正向化后的矩阵
        """
        self.M = matrix
        self.P = None
        """评价指标矩阵"""
        self.E = None
        """熵值向量"""
        self.W = None
        """权重向量"""

    def normalize(self, stand_func=prob):
        """
        计算指标判断矩阵的占比
        """
        P = stand_func(self.M)
        self.P = P

    def calc_entropy(self, offset=1):
        """计算熵权"""
        E = -np.sum(self.P * np.log(self.P + offset), axis=0) / np.log(self.m)
        self.E = E

    def calc_weight(self, norm_func=prob):
        """计算权重"""
        W = norm_func(1 - self.E)
        self.W = W

    def run(self):
        self.normalize()
        self.calc_entropy()
        self.calc_weight()
        return self.W


if __name__ == "__main__":
    M = np.array(
        [
            [0.5, 0.4, 0.3, 0.2],
            [0.4, 0.5, 0.3, 0.2],
            [0.3, 0.3, 0.5, 0.2],
            [0.2, 0.2, 0.2, 0.5],
        ]
    )
    ewm = EWM(M)
    W = ewm.run()
    print(W)
