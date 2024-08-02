"""距离函数库"""

import numpy as np

from .type_ import ArrayLike

__all__ = ["euclidean", "cosine", "manhattan"]


def euclidean(x: ArrayLike, y: ArrayLike) -> float:
    """计算欧氏距离"""
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x - y)


def cosine(x: ArrayLike, y: ArrayLike) -> float:
    """计算余弦相似度，然后转换为余弦距离"""
    x = np.array(x)
    y = np.array(y)
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 or norm_y == 0:
        raise ValueError(
            "One of the vectors is a zero vector, cannot compute cosine similarity."
        )
    cosine_similarity = dot_product / (norm_x * norm_y)
    return 1 - cosine_similarity


def manhattan(x: ArrayLike, y: ArrayLike) -> float:
    """计算曼哈顿距离"""
    x = np.array(x)
    y = np.array(y)
    return np.sum(np.abs(x - y))
