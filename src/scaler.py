import numpy as np

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
