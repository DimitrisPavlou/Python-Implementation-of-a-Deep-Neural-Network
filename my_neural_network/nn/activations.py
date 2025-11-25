
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings
from abc import ABC, abstractmethod
from .core import Activation


class ReLU(Activation):
    """ReLU activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -500, 500)  # Prevent overflow
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.inputs > 0)

class LeakyReLU(Activation):
    """Leaky ReLU activation function."""
    
    def __init__(self, negative_slope: float = 0.01):
        if negative_slope < 0:
            raise ValueError(f"negative_slope must be non-negative. Got: {negative_slope}")
        self.negative_slope = negative_slope
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -500, 500)
        self.inputs = inputs
        return np.where(inputs > 0, inputs, self.negative_slope * inputs)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * np.where(self.inputs > 0, 1, self.negative_slope)

class SiLU(Activation):
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -10, 10)
        self.inputs = inputs
        self.sigmoid = 1 / (1 + np.exp(-inputs))
        return inputs * self.sigmoid
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        silu_derivative = self.sigmoid + self.inputs * self.sigmoid * (1 - self.sigmoid)
        return grad_output * silu_derivative

class GELU(Activation):
    """GELU activation function (approximation)."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -10, 10)
        self.inputs = inputs
        # Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        tanh_arg = np.sqrt(2 / np.pi) * (inputs + 0.044715 * inputs**3)
        self.tanh_val = np.tanh(tanh_arg)
        return 0.5 * inputs * (1 + self.tanh_val)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Derivative of GELU approximation
        tanh_arg = np.sqrt(2 / np.pi) * (self.inputs + 0.044715 * self.inputs**3)
        sech2 = 1 - self.tanh_val**2
        
        gelu_derivative = 0.5 * (1 + self.tanh_val) + \
                         0.5 * self.inputs * sech2 * np.sqrt(2 / np.pi) * \
                         (1 + 0.134145 * self.inputs**2)
        
        return grad_output * gelu_derivative

class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -10, 10)
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.output * (1 - self.output)

class Tanh(Activation):
    """Tanh activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -10, 10)
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1 - self.output**2)

class Softmax(Activation):
    """Softmax activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        # Subtract max for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = np.empty_like(grad_output)
        
        for i, (single_output, single_grad) in enumerate(zip(self.output, grad_output)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            grad_input[i] = np.dot(jacobian, single_grad)
        
        return grad_input