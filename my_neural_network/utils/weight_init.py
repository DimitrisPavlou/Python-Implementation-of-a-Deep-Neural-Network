import numpy as np
from typing import Tuple

class WeightInitializer:
    """Weight initialization strategies."""
    
    @staticmethod
    def xavier_uniform(shape: Tuple[int, ...]) -> np.ndarray:
        """Xavier/Glorot uniform initialization."""
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def xavier_normal(shape: Tuple[int, ...]) -> np.ndarray:
        """Xavier/Glorot normal initialization."""
        fan_in, fan_out = shape[0], shape[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)
    
    @staticmethod
    def he_uniform(shape: Tuple[int, ...]) -> np.ndarray:
        """He uniform initialization (good for ReLU)."""
        fan_in = shape[0]
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def he_normal(shape: Tuple[int, ...]) -> np.ndarray:
        """He normal initialization (good for ReLU)."""
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape)