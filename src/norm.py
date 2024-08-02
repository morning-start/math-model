"""
归一化函数库
"""

import numpy as np

from src.type_ import ArrayLike


def softmax(x: ArrayLike) -> np.ndarray:
    """Softmax函数"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def prob(data: ArrayLike) -> np.ndarray:
    """概率归一化"""
    prob_sum = np.sum(data, axis=0)
    return data / prob_sum
