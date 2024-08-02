"""
归一化函数库
"""

import numpy as np
from scipy.spatial.distance import euclidean

from src.type_ import ArrayLike


def min_max_scaler(data: ArrayLike, min_val=0.0, max_val=1.0) -> np.ndarray:
    """最大最小归一化（Min-Max Scaling）"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data * (max_val - min_val) + min_val


def z_score(data: ArrayLike) -> np.ndarray:
    """Z-Score Normalization（标准化）"""
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev

    return normalized_data


def softmax(x: ArrayLike) -> np.ndarray:
    """Softmax函数"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def prob(data: ArrayLike) -> np.ndarray:
    """概率归一化"""
    prob_sum = np.sum(data, axis=0)
    return data / prob_sum


if __name__ == "__main__":
    import os
    import sys

    # cur_dir = os.getcwd()
    # root_dir = os.path.dirname(cur_dir)
    # if cur_dir not in sys.path:
    #     sys.path.extend([cur_dir, root_dir])

    print(sys.path)
