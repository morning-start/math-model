import numpy as np

from .. import distance, norm


class TOPSIS:
    """TOPSIS法可翻译为逼近理想解排序法，国内常简称为优劣解距离法。"""

    def __init__(self, matrix: np.ndarray, weights: list, signs: list[bool]):
        """
        Parameters
        ---
        matrix: np.ndarray
            正向化后的矩阵
        weights: list
            每个指标的权重
        signs: list[bool]
            每个指标的类别，其中：
            - true: 效益指标
            - false: 成本指标
        """
        self.matrix = np.array(matrix)
        self.weights = np.array(weights)
        self.signs = np.array(signs)
        self.n = self.matrix.shape[0]  # 备选方案数
        self.m = self.matrix.shape[1]  # 准则数
        self.ideal_solution: tuple = None
        """理想解，(positive, negative)"""

    def normalize(self, norm_func=norm.euclidean):
        """标准化决策矩阵"""
        self.Z = norm_func(self.matrix)

    def weighed_matrix(self):
        """计算加权后的决策矩阵"""
        self.weighted_matrix = self.Z * self.weights.reshape(-1, 1)

    def calc_ideal_solution(self):
        """计算理想解和负理想解"""
        positive_solution = np.max(self.weighted_matrix * self.signs, axis=0)
        negative_solution = np.min(self.weighted_matrix * ~self.signs, axis=0)
        self.ideal_solution = (positive_solution, negative_solution)

    def calc_distance(self, distance_func=distance.euclidean):
        """计算每个方案与理想解和负理想解的距离"""
        positive_solution, negative_solution = self.ideal_solution
        distance_positive = distance_func(self.weighted_matrix, positive_solution, 1)
        distance_negative = distance_func(self.weighted_matrix, negative_solution, 1)
        self.distance_ideal = (distance_positive, distance_negative)
        print(distance_positive, distance_negative)

    def calc_closeness(self):
        """计算相对贴近度"""
        distance_positive, distance_negative = self.distance_ideal
        closeness = distance_negative / (distance_positive + distance_negative)
        self.closeness = closeness

    def rank(self):
        """根据相对贴近度进行排序"""
        closeness = self.closeness
        ranking = np.argsort(-closeness)  # 贴近度越大排名越前
        self.ranking = ranking
        return ranking

    def run(self):
        self.normalize()
        self.weighed_matrix()
        self.calc_ideal_solution()
        self.calc_distance()
        self.calc_closeness()
        return self.rank()


if __name__ == "__main__":

    # 示例使用
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    weights = [0.5, 0.3, 0.2]  # 权重示例
    signs = [True, True, False]  # 指标类型，True为效益型，False为成本型

    topsis = TOPSIS(matrix, weights, signs)
    ranking = topsis.run()
    print("方案排名:", ranking)
