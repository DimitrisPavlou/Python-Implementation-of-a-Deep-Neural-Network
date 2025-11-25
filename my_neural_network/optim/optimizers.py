"""
Optimizers Module

Contains various optimization algorithms for neural network training.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
# ==============================================================================
# BASE OPTIMIZER CLASS
# ==============================================================================

class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.001):
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive. Got: {learning_rate}")
        
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.iterations = 0
    
    @abstractmethod
    def update_params(self, layer) -> None:
        """Update layer parameters."""
        pass
    
    def pre_update(self) -> None:
        """Called before parameter updates."""
        pass
    
    def post_update(self) -> None:
        """Called after parameter updates."""
        self.iterations += 1

# ==============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ==============================================================================

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, 
                 lr_decay: float = 0.0, momentum_decay: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            lr_decay: Learning rate decay factor
            momentum_decay: Momentum decay factor
        """
        super().__init__(learning_rate)
        
        if momentum < 0 or momentum > 1:
            raise ValueError(f"Momentum must be in [0, 1]. Got: {momentum}")
        
        if lr_decay < 0:
            raise ValueError(f"Learning rate decay must be non-negative. Got: {lr_decay}")
            
        if momentum_decay < 0:
            raise ValueError(f"Momentum decay must be non-negative. Got: {momentum_decay}")
        
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.momentum_decay = momentum_decay
        self.current_momentum = momentum
    
    def pre_update(self) -> None:
        """Apply learning rate and momentum decay."""
        # Learning rate decay
        if self.lr_decay > 0:
            self.current_lr = self.learning_rate / (1 + self.lr_decay * self.iterations)
            # Minimum learning rate
            self.current_lr = max(self.current_lr, 1e-7)
        
        # Momentum decay
        if self.momentum_decay > 0:
            self.current_momentum = self.momentum * (1 - self.momentum_decay * self.iterations)
            self.current_momentum = max(self.current_momentum, 0.0)
    
    def update_params(self, layer) -> None:
        """Update layer parameters using SGD with momentum."""
        if not hasattr(layer, 'parameters') or not hasattr(layer, 'gradients'):
            return
        
        for param_name, param_value in layer.parameters.items():
            if layer.gradients[param_name] is None:
                continue
            
            momentum_attr = f'{param_name}_momentum'
            
            if self.current_momentum > 0:
                # Get or initialize momentum
                if not hasattr(layer, momentum_attr):
                    setattr(layer, momentum_attr, np.zeros_like(param_value))
                
                momentum_value = getattr(layer, momentum_attr)
                
                # Update momentum
                update = self.current_momentum * momentum_value - self.current_lr * layer.gradients[param_name]
                setattr(layer, momentum_attr, update)
            else:
                # Standard SGD update
                update = -self.current_lr * layer.gradients[param_name]
            
            # Update parameters
            param_value += update

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-7, decay: float = 0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            decay: Learning rate decay
        """
        super().__init__(learning_rate)
        
        if not 0 <= beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1). Got: {beta1}")
        
        if not 0 <= beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1). Got: {beta2}")
        
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive. Got: {epsilon}")
        
        if decay < 0:
            raise ValueError(f"decay must be non-negative. Got: {decay}")
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
    
    def pre_update(self) -> None:
        """Apply learning rate decay."""
        if self.decay > 0:
            self.current_lr = self.learning_rate / (1 + self.decay * self.iterations)
            self.current_lr = max(self.current_lr, 1e-7)
    
    def update_params(self, layer) -> None:
        """Update layer parameters using Adam."""
        if not hasattr(layer, 'parameters') or not hasattr(layer, 'gradients'):
            return
        
        for param_name, param_value in layer.parameters.items():
            if layer.gradients[param_name] is None:
                continue
            
            momentum_attr = f'{param_name}_momentum'
            cache_attr = f'{param_name}_cache'
            
            # Initialize momentum and cache if needed
            if not hasattr(layer, momentum_attr):
                setattr(layer, momentum_attr, np.zeros_like(param_value))
            if not hasattr(layer, cache_attr):
                setattr(layer, cache_attr, np.zeros_like(param_value))
            
            momentum = getattr(layer, momentum_attr)
            cache = getattr(layer, cache_attr)
            
            # Update biased first moment estimate
            momentum *= self.beta1
            momentum += (1 - self.beta1) * layer.gradients[param_name]
            
            # Update biased second raw moment estimate
            cache *= self.beta2
            cache += (1 - self.beta2) * layer.gradients[param_name]**2
            
            # Bias correction
            momentum_corrected = momentum / (1 - self.beta1**(self.iterations + 1))
            cache_corrected = cache / (1 - self.beta2**(self.iterations + 1))
            
            # Update parameters
            param_value -= self.current_lr * momentum_corrected / (np.sqrt(cache_corrected) + self.epsilon)
            
            # Store updated momentum and cache
            setattr(layer, momentum_attr, momentum)
            setattr(layer, cache_attr, cache)

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-7, 
                 weight_decay: float = 0.01, decay: float = 0.0):
        """
        Initialize AdamW optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay coefficient
            decay: Learning rate decay
        """
        super().__init__(learning_rate)
        
        if not 0 <= beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1). Got: {beta1}")
        
        if not 0 <= beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1). Got: {beta2}")
        
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive. Got: {epsilon}")
        
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative. Got: {weight_decay}")
        
        if decay < 0:
            raise ValueError(f"decay must be non-negative. Got: {decay}")
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.decay = decay
    
    def pre_update(self) -> None:
        """Apply learning rate decay."""
        if self.decay > 0:
            self.current_lr = self.learning_rate / (1 + self.decay * self.iterations)
            self.current_lr = max(self.current_lr, 1e-7)
    
    def update_params(self, layer) -> None:
        """Update layer parameters using AdamW."""
        if not hasattr(layer, 'parameters') or not hasattr(layer, 'gradients'):
            return
        
        for param_name, param_value in layer.parameters.items():
            if layer.gradients[param_name] is None:
                continue
            
            momentum_attr = f'{param_name}_momentum'
            cache_attr = f'{param_name}_cache'
            
            # Initialize momentum and cache if needed
            if not hasattr(layer, momentum_attr):
                setattr(layer, momentum_attr, np.zeros_like(param_value))
            if not hasattr(layer, cache_attr):
                setattr(layer, cache_attr, np.zeros_like(param_value))
            
            momentum = getattr(layer, momentum_attr)
            cache = getattr(layer, cache_attr)
            
            # Update biased first moment estimate
            momentum *= self.beta1
            momentum += (1 - self.beta1) * layer.gradients[param_name]
            
            # Update biased second raw moment estimate
            cache *= self.beta2
            cache += (1 - self.beta2) * layer.gradients[param_name]**2
            
            # Bias correction
            momentum_corrected = momentum / (1 - self.beta1**(self.iterations + 1))
            cache_corrected = cache / (1 - self.beta2**(self.iterations + 1))
            
            # AdamW: Apply weight decay directly to parameters (not to gradients)
            if self.weight_decay > 0 and 'weight' in param_name:
                param_value *= (1 - self.current_lr * self.weight_decay)
            
            # Update parameters
            param_value -= self.current_lr * momentum_corrected / (np.sqrt(cache_corrected) + self.epsilon)
            
            # Store updated momentum and cache
            setattr(layer, momentum_attr, momentum)
            setattr(layer, cache_attr, cache)

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, alpha: float = 0.99, 
                 epsilon: float = 1e-7, decay: float = 0.0):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            alpha: Smoothing constant
            epsilon: Small constant for numerical stability
            decay: Learning rate decay
        """
        super().__init__(learning_rate)
        
        if not 0 <= alpha < 1:
            raise ValueError(f"alpha must be in [0, 1). Got: {alpha}")
        
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive. Got: {epsilon}")
        
        if decay < 0:
            raise ValueError(f"decay must be non-negative. Got: {decay}")
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
    
    def pre_update(self) -> None:
        """Apply learning rate decay."""
        if self.decay > 0:
            self.current_lr = self.learning_rate / (1 + self.decay * self.iterations)
            self.current_lr = max(self.current_lr, 1e-7)
    
    def update_params(self, layer) -> None:
        """Update layer parameters using RMSprop."""
        if not hasattr(layer, 'parameters') or not hasattr(layer, 'gradients'):
            return
        
        for param_name, param_value in layer.parameters.items():
            if layer.gradients[param_name] is None:
                continue
            
            cache_attr = f'{param_name}_cache'
            
            # Initialize cache if needed
            if not hasattr(layer, cache_attr):
                setattr(layer, cache_attr, np.zeros_like(param_value))
            
            cache = getattr(layer, cache_attr)
            
            # Update cache
            cache *= self.alpha
            cache += (1 - self.alpha) * layer.gradients[param_name]**2
            
            # Update parameters
            param_value -= self.current_lr * layer.gradients[param_name] / (np.sqrt(cache) + self.epsilon)
            
            # Store updated cache
            setattr(layer, cache_attr, cache)



class LRScheduler:
    """Learning rate scheduler."""
    
    def __init__(self, optimizer, mode: str = 'step', **kwargs):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: Scheduling mode ('step', 'cosine', 'plateau')
            **kwargs: Mode-specific parameters
        """
        self.optimizer = optimizer
        self.mode = mode
        self.initial_lr = optimizer.learning_rate
        
        if mode == 'step':
            self.step_size = kwargs.get('step_size', 10)
            self.gamma = kwargs.get('gamma', 0.1)
        elif mode == 'cosine':
            self.T_max = kwargs.get('T_max', 100)
            self.eta_min = kwargs.get('eta_min', 0)
        elif mode == 'plateau':
            self.patience = kwargs.get('patience', 10)
            self.factor = kwargs.get('factor', 0.1)
            self.threshold = kwargs.get('threshold', 1e-4)
            self.best_score = None
            self.wait = 0
    
    def step(self, epoch: int, metric: Optional[float] = None):
        """Update learning rate."""
        if self.mode == 'step':
            if epoch % self.step_size == 0 and epoch > 0:
                self.optimizer.current_lr *= self.gamma
        
        elif self.mode == 'cosine':
            self.optimizer.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                                       (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        
        elif self.mode == 'plateau':
            if metric is None:
                raise ValueError("Metric required for plateau scheduler")
            
            if self.best_score is None or metric < (self.best_score - self.threshold):
                self.best_score = metric
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.optimizer.current_lr *= self.factor
                    self.wait = 0



# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_optimizer(name: str, **kwargs) -> Optimizer:
    """Factory function to get optimizer by name."""
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop
    }
    
    if name.lower() not in optimizers:
        available = ', '.join(optimizers.keys())
        raise ValueError(f"Unknown optimizer: {name}. Available: {available}")
    
    return optimizers[name.lower()](**kwargs)