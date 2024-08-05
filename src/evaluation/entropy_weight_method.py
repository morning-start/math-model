import numpy as np

from src.norm import min_max, prob


class EWM:
    """
    熵权法（Entropy Weight Method, EWM）是一种客观赋权方法，基于信息论的理论基础，
    根据各指标的数据的分散程度，利用信息熵计算并修正得到各指标的熵权，较为客观。
    相对而言这种数据驱动的方法就回避了上面主观性因素造成的重复修正的影响。
    """

    def __init__(self, matrix: np.ndarray):
        self.M = matrix
        self.m, self.n = matrix.shape
        self.P = None
        """评价指标矩阵"""
        self.E = None
        """熵值向量"""
        self.W = None
        """权重向量"""

    def judgment_matrix_proportion(self):
        """
        计算指标判断矩阵的占比
        """
        X = min_max(self.M)
        P = prob(X)
        self.P = P

    def calculate_entropy_weight(self, offset=1):
        """计算熵权"""
        P = self.P + offset
        E = -np.sum(P * np.log(P), axis=0) / np.log(self.m)
        self.E = E

    def calculate_weight(self):
        """计算权重"""
        W = prob(1 - self.E)
        self.W = W

    def run(self):
        self.judgment_matrix_proportion()
        self.calculate_entropy_weight()
        self.calculate_weight()
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
