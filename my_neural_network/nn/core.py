import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Abstract base class for all neural network layers."""
    
    def __init__(self):
        self.training = True
        self.parameters = {}
        self.gradients = {}
        
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through the layer."""
        pass
    
    def train(self):
        """Set layer to training mode."""
        self.training = True
        
    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False

class Activation(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass

class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        pass