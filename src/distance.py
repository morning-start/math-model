"""距离函数库"""

import numpy as np

from .type_ import ArrayLike

__all__ = ["euclidean", "cosine", "manhattan"]


def euclidean(x: ArrayLike, y: ArrayLike, axis=0) -> float | ArrayLike:
    """
    计算两个数组之间的欧氏距离。

    Parameters
    ----------
    x : ArrayLike
        输入数组x，可以是列表或numpy数组。
    y : ArrayLike
        输入数组y，可以是列表或numpy数组。
    axis : int, optional
        指定沿哪个轴计算距离，默认为0（即沿着列方向）。

    Returns
    -------
    float
        返回计算得到的欧氏距离。
    """
    x = np.array(x)
    y = np.array(y)

    return np.linalg.norm(x - y, axis=axis)


def cosine(x: ArrayLike, y: ArrayLike, axis: int = 0) -> float | ArrayLike:
    """
    计算两个数组的余弦相似度，并将其转换为余弦距离。

    Parameters
    ----------
    x : ArrayLike
        输入数组x，可以是列表或numpy数组。
    y : ArrayLike
        输入数组y，可以是列表或numpy数组。
    axis : int, optional
        指定沿哪个轴计算距离，默认为0（即沿着列方向）。

    Returns
    -------
    float
        返回计算得到的余弦距离。
    """
    # 将输入转换为numpy数组
    x = np.asarray(x)
    y = np.asarray(y)

    # 检查x和y的形状是否兼容，用于计算点积
    if x.shape != y.shape:
        raise ValueError("输入数组的形状必须相同以计算余弦相似度")

    # 计算点积
    dot_product = np.sum(x * y, axis=axis)

    # 计算x和y的范数
    norm_x = np.linalg.norm(x, axis=axis)
    norm_y = np.linalg.norm(y, axis=axis)

    # 避免除以零的情况
    if norm_x == 0 or norm_y == 0:
        raise ValueError(
            "One of the vectors is a zero vector, cannot compute cosine similarity."
        )

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_x * norm_y)

    # 将余弦相似度转换为余弦距离
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


def manhattan(x: ArrayLike, y: ArrayLike, axis=0) -> float | ArrayLike:
    """
    计算曼哈顿距离

    Parameters
    ----------
    x : ArrayLike
        输入数组x，可以是列表或numpy数组。
    y : ArrayLike
        输入数组y，可以是列表或numpy数组。
    axis : int, optional
        指定沿哪个轴计算距离，默认为0（即沿着列方向）。

    Returns
    -------
    float
        返回计算得到的曼哈顿距离。

    """
    x = np.array(x)
    y = np.array(y)
    return np.sum(np.abs(x - y), axis=axis)
