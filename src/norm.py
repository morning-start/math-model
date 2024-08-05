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


def min_max(data: ArrayLike) -> np.ndarray:
    """最大最小归一化（Min-Max Scaling）"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data


def z_score(data: ArrayLike) -> np.ndarray:
    """Z-Score Normalization（标准化）"""
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev

    return normalized_data


def euclidean(matrix):
    """
    对输入矩阵进行欧氏范数归一化处理。

    Parameters:
    ---
    matrix: np.ndarray
        需要归一化的原始数据矩阵。

    Returns:
    ---
    normalized_matrix: np.ndarray
        归一化后的数据矩阵。
    """
    normalized_matrix = matrix / np.linalg.norm(matrix, axis=0)

    return normalized_matrix
