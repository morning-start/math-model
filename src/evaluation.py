"""评价模型库"""

import itertools
from dataclasses import dataclass, field
from functools import reduce

import numpy as np

from src.norm import prob as norm_func


class AHP:
    """
    层次分析法（Analytic Hierarchy Process, AHP）是由美国运筹学家托马斯·L·萨蒂（T.L. Saaty）
    在20世纪70年代初期提出的，用于解决多目标的复杂问题的定性与定量相结合的决策分析方法

    1. **选择指标**，构建层次模型，并构建**比较矩阵**。
    2. 对每个比较矩阵计算CR值检验是否通过CR检验，**如果没有通过检验需要调整比较矩阵**。
    3. 通过比较矩阵计算权重
    4. 根据权重向量计算得分，并排序。

    内部类主要 Criterion 主要实现了 AHP 算法的核心，1-3 的步骤。
    外部类 AHP 主要实现了整体流程，1、4 步骤
    """

    @dataclass
    class Criterion:
        """
        一个比较矩阵的信息，实现了 AHP 算法的核心

        Note
        ----
        如果是改进的AHP算法可以通过以下方法

        1. 或者替换掉计算权重的方法，传入参数就是 self 如：
        >>> AHP.Criterion.calculate = new_func
        >>> AHP.Criterion.calculate = lambda self: new_func(self.matrix)

        2. 或者使用代理的方式，或装饰器方式如：

        3. 继承AHP类然后继承内部类（不建议）
        """

        level: str
        """指标等级: A1, A2,... B1, B2,..."""
        matrix: np.ndarray
        """判断矩阵"""
        W: np.ndarray = field(init=False)
        """归一化后的权重向量"""
        CR: float = field(init=False)
        """一致性比率，小于 0.1 通过检验"""

        def __post_init__(self):
            self.matrix = self.check_matrix()
            self.CR, self.W = self.calculate()

        def check_matrix(self):
            """
            检查判断矩阵是否符合性质
            """
            matrix = np.array(self.matrix)
            bool_M: np.ndarray = matrix.T * matrix == np.ones_like(matrix)
            flag = bool_M.all()
            if flag:
                return matrix
            else:
                raise ValueError("The matrix is not a valid matrix.")

        def calculate(self) -> tuple[float, np.ndarray]:
            """
            一致性检验+计算权重

            Returns
            ---
            CR: float
                一致性比率。
            W: np.ndarray
                归一化后的权重向量。
            """
            V, D = np.linalg.eig(self.matrix)
            v, d = V[0], D.T[0]
            # 都只取实数部分
            v, d = v.real, d.real
            n = self.matrix.shape[0]
            CI = (v - n) / (n - 1)
            if n == 1:
                return 0, np.ones_like(self.matrix)
            elif n == 2:
                return 0, norm_func(d)
            else:
                RI = AHP.get_RI(n)
                CR = CI / RI
                W = norm_func(d)
            return CR, W

        def check_CR(self, threshold=0.1):
            """检查一致性比率, 是否小于 0.1"""
            CR = self.CR
            return CR < threshold

        def __str__(self) -> str:
            return f"{self.level}"

        def __repr__(self) -> str:
            return f"{self.level}"

        def __len__(self):
            return self.matrix.shape[0]

    RI_map = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    @classmethod
    def get_RI(cls, n: int) -> float:
        """
        获取RI值。

        Parameters
        ---
        n: int
            比较矩阵的维度。

        Returns
        ---
        RI: float
            RI值。

        """
        RI = AHP.RI_map.get(n, 1.5)
        return RI

    def __init__(self, criterions: list[Criterion]):
        """
        初始化AHP类。

        Parameters
        ---
        criterions: list[Criterion]
            比较矩阵列表。
        """
        self.criterions = sorted(criterions, key=lambda c: c.level)
        """比较矩阵列表"""
        self.grouped = AHP.grouped_by_level(criterions)
        """分组后的比较矩阵，分组前进行CR检验"""
        self.W = None
        """AHP综合权重向量"""

    @staticmethod
    def grouped_by_level(criterions: list[Criterion]):
        """分组后的比较矩阵，分组前进行CR检验"""
        group = {
            k: [x.W for x in sorted(g, key=lambda x: x.level)]
            for k, g in itertools.groupby(criterions, lambda e: e.level[0])
        }
        # 按照level属性排序
        group = {k: v for k, v in sorted(group.items(), key=lambda e: e[0])}
        return group

    def run(self, threshold=0.1) -> list[Criterion]:
        """融合权重并排序"""
        # 判断是否有一致性不合格的矩阵
        if any(not c.check_CR(threshold) for c in self.criterions):
            raise ValueError("Comparison matrix CR is not pass the CR test.")
        # 计算综合权重
        group_pad: dict[str, np.ndarray] = self.grouped
        max_len = max(len(x) for x in self.criterions)
        for k, v in group_pad.items():
            padded_v = [np.pad(vec, (0, max_len - len(vec)), "constant") for vec in v]
            group_pad[k] = np.array(padded_v)

        def multi_layer_weight(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """融合层间权重向量"""
            x = x.reshape(-1, 1)
            z = x * y
            z = z.flatten()
            return z[z != 0]

        w = reduce(multi_layer_weight, group_pad.values())
        self.W = w


if __name__ == "__main__":
    B = [
        [1, 7, 5, 7, 5],
        [1 / 7, 1, 2, 3, 3],
        [1 / 5, 1 / 2, 1, 2, 3],
        [1 / 7, 1 / 3, 1 / 2, 1, 3],
        [1 / 5, 1 / 3, 1 / 3, 1 / 3, 1],
    ]
    C1 = [[1, 5], [1 / 5, 1]]
    C2 = [[1, 2, 5], [1 / 2, 1, 2], [1 / 5, 1 / 2, 1]]
    C3 = [
        [1, 5, 6, 8],
        [1 / 5, 1, 2, 7],
        [1 / 6, 1 / 2, 1, 4],
        [1 / 8, 1 / 7, 1 / 4, 1],
    ]
    C4 = [[1, 3, 4], [1 / 3, 1, 1], [1 / 4, 1, 1]]
    C5 = [
        [1, 4, 5, 5],
        [1 / 4, 1, 2, 4],
        [1 / 5, 1 / 2, 1, 2],
        [1 / 5, 1 / 4, 1 / 2, 1],
    ]

    B = AHP.Criterion("B", B)
    C1 = AHP.Criterion("C1", C1)
    C2 = AHP.Criterion("C2", C2)
    C3 = AHP.Criterion("C3", C3)
    C4 = AHP.Criterion("C4", C4)
    C5 = AHP.Criterion("C5", C5)
    arr = [C1, C2, C3, C4, C5, B]
    ahp = AHP(arr)
    ahp.run()
