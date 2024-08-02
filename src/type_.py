from typing import Callable

import numpy as np

DistanceFuncType = Callable[[np.ndarray, np.ndarray], float]
WeightFuncType = Callable[[np.ndarray], np.ndarray]
ArrayLike = list | tuple | np.ndarray
