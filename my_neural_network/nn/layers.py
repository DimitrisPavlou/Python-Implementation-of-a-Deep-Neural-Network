import numpy as np
from .core import Layer
from ..utils.weight_init import WeightInitializer


class Linear(Layer):
    """Linear (Dense) layer with proper initialization."""
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, init_method: str = 'he_normal'):
        """
        Initialize Linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features  
            bias: Whether to include bias term
            init_method: Weight initialization method
        """
        super().__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"Features must be positive integers. Got in_features={in_features}, out_features={out_features}")
        
        if init_method not in ['xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal']:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.init_method = init_method
        
        # Initialize weights
        init_func = getattr(WeightInitializer, init_method)
        self.weights = init_func((in_features, out_features))
        
        if bias:
            self.bias = np.zeros((1, out_features))
        else:
            self.bias = None
            
        # Initialize momentum and cache for optimizers
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias) if bias else None
        self.weight_cache = np.zeros_like(self.weights)
        self.bias_cache = np.zeros_like(self.bias) if bias else None
        
        # Store parameters and gradients
        self.parameters = {'weights': self.weights}
        self.gradients = {'weights': None}
        
        if bias:
            self.parameters['bias'] = self.bias
            self.gradients['bias'] = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if inputs.ndim != 2:
            raise ValueError(f"Input must be 2D array. Got shape: {inputs.shape}")
        
        if inputs.shape[1] != self.in_features:
            raise ValueError(f"Input features ({inputs.shape[1]}) don't match layer input features ({self.in_features})")
        
        self.inputs = inputs
        output = np.dot(inputs, self.weights)
        
        if self.use_bias:
            output += self.bias
            
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        if grad_output.shape[1] != self.out_features:
            raise ValueError(f"Gradient output shape mismatch. Expected {self.out_features}, got {grad_output.shape[1]}")
        
        # Compute gradients
        self.gradients['weights'] = np.dot(self.inputs.T, grad_output)
        
        if self.use_bias:
            self.gradients['bias'] = np.sum(grad_output, axis=0, keepdims=True)
        
        # Compute gradient w.r.t. inputs
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input

class BatchNorm1d(Layer):
    """1D Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize Batch Normalization layer.
        
        Args:
            num_features: Number of features
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        
        if num_features <= 0:
            raise ValueError(f"num_features must be positive. Got: {num_features}")
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))  # Scale
        self.beta = np.zeros((1, num_features))  # Shift
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Store for backward pass
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        
        self.parameters = {'gamma': self.gamma, 'beta': self.beta}
        self.gradients = {'gamma': None, 'beta': None}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if inputs.shape[1] != self.num_features:
            raise ValueError(f"Input features ({inputs.shape[1]}) don't match num_features ({self.num_features})")
        
        self.inputs = inputs
        
        if self.training:
            # Compute batch statistics
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
            
            # Normalize
            self.normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
        else:
            # Use running statistics
            self.normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        output = self.gamma * self.normalized + self.beta
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        batch_size = self.inputs.shape[0]
        
        # Gradients w.r.t. gamma and beta
        self.gradients['gamma'] = np.sum(grad_output * self.normalized, axis=0, keepdims=True)
        self.gradients['beta'] = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. normalized input
        grad_normalized = grad_output * self.gamma
        
        # Gradient w.r.t. variance
        grad_var = np.sum(grad_normalized * (self.inputs - self.batch_mean) * 
                         -0.5 * (self.batch_var + self.eps) ** -1.5, axis=0, keepdims=True)
        
        # Gradient w.r.t. mean
        grad_mean = np.sum(grad_normalized * -1.0 / np.sqrt(self.batch_var + self.eps), axis=0, keepdims=True) + \
                   grad_var * np.sum(-2.0 * (self.inputs - self.batch_mean), axis=0, keepdims=True) / batch_size
        
        # Gradient w.r.t. input
        grad_input = grad_normalized / np.sqrt(self.batch_var + self.eps) + \
                    grad_var * 2.0 * (self.inputs - self.batch_mean) / batch_size + \
                    grad_mean / batch_size
        
        return grad_input