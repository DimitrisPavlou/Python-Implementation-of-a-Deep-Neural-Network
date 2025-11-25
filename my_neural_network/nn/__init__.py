"""
Neural Network Module - Contains layers, activations, and loss functions
"""

# Import from modules
from .core import (
    Layer, Loss, Activation
)

from .layers import (
    Linear, BatchNorm1d
)

from .activations import (
    ReLU, LeakyReLU, Sigmoid, SiLU, GELU, Softmax, Tanh, 
)

# Import from functional
from .loss_functions import (
    MSELoss, SmoothL1Loss, SoftmaxCrossEntropyLoss, CrossEntropyLoss
)

# Import specific classes for direct access
# Define what gets imported with "from my_neural_network.nn import *"
__all__ = [
    # Base classes
    'Layer', 'Activation', 'Loss',
    
    # Layers
    'Linear', 'BatchNorm1d',
    
    # Activation classes
    'ReLU', 'LeakyReLU', 'SiLU', 'GELU',
    'Sigmoid', 'Tanh', 'Softmax',
    
    # Loss classes
    'MSELoss', 'SmoothL1Loss', 'CrossEntropyLoss', 'SoftmaxCrossEntropyLoss',
]

## Functional API
#    'relu', 'leaky_relu', 'silu', 'gelu',
#    'sigmoid', 'tanh', 'softmax',
#    'mse_loss', 'smooth_l1_loss', 'cross_entropy_loss', 'softmax_cross_entropy_loss'