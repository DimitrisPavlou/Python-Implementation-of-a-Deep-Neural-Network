"""
Loss Functions Module

Contains various loss functions for neural network training.
"""
import numpy as np
from .core import Loss 
from .activations import Softmax

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class MSELoss(Loss):
    """Mean Squared Error Loss."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        self.predictions = predictions
        self.targets = targets
        
        sample_losses = 0.5 * np.sum((targets - predictions)**2, axis=-1)
        return np.mean(sample_losses)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute MSE gradient."""
        samples, outputs = predictions.shape
        grad = -(targets - predictions) / outputs
        return grad / samples

class SmoothL1Loss(Loss):
    """Smooth L1 Loss (Huber Loss with delta=1)."""
    
    def __init__(self, beta: float = 1.0):
        """
        Initialize Smooth L1 Loss.
        
        Args:
            beta: Threshold for switching between L1 and L2 loss
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive. Got: {beta}")
        self.beta = beta
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Smooth L1 loss."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        self.predictions = predictions
        self.targets = targets
        
        diff = np.abs(targets - predictions)
        
        # Smooth L1: 0.5 * x^2 / beta if |x| < beta, else |x| - 0.5 * beta
        loss = np.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        return np.mean(np.sum(loss, axis=-1))
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute Smooth L1 gradient."""
        samples = predictions.shape[0]
        diff = predictions - targets
        
        # Gradient: x / beta if |x| < beta, else sign(x)
        grad = np.where(
            np.abs(diff) < self.beta,
            diff / self.beta,
            np.sign(diff)
        )
        
        return grad / samples

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss for multi-class classification."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross entropy loss."""
        samples = predictions.shape[0]
        
        # Clip predictions to prevent log(0)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        self.predictions = predictions_clipped
        self.targets = targets
        
        # Handle both sparse (integer) and one-hot encoded targets
        if targets.ndim == 1:
            # Sparse targets
            correct_confidences = predictions_clipped[range(samples), targets]
        else:
            # One-hot encoded targets
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute cross entropy gradient."""
        samples = predictions.shape[0]
        labels = predictions.shape[1]
        
        # Convert sparse targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = np.eye(labels)[targets]
        else:
            targets_one_hot = targets
        
        # Gradient
        grad = -targets_one_hot / predictions
        return grad / samples

class SoftmaxCrossEntropyLoss(Loss):
    """Combined Softmax and Cross Entropy for efficiency."""
    
    def __init__(self):
        self.softmax = Softmax()
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Forward pass through softmax and cross entropy."""
        # Apply softmax
        self.predictions = self.softmax.forward(logits)
        
        # Compute cross entropy
        samples = logits.shape[0]
        
        self.targets = targets
        
        # Handle both sparse and one-hot targets
        if targets.ndim == 1:
            correct_confidences = self.predictions[range(samples), targets]
        else:
            correct_confidences = np.sum(self.predictions * targets, axis=1)
        
        negative_log_likelihoods = -np.log(np.clip(correct_confidences, 1e-7, 1.0))
        return np.mean(negative_log_likelihoods)
    
    def backward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Backward pass - simplified gradient for softmax + cross entropy."""
        samples = logits.shape[0]
        
        # Convert sparse targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = np.eye(logits.shape[1])[targets]
        else:
            targets_one_hot = targets
        
        # Simplified gradient: softmax_output - targets
        grad = self.predictions - targets_one_hot
        return grad / samples

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_loss_function(name: str, **kwargs) -> Loss:
    """Factory function to get loss function by name."""
    loss_functions = {
        'mse': MSELoss,
        'smooth_l1': SmoothL1Loss,
        'cross_entropy': CrossEntropyLoss,
        'softmax_cross_entropy': SoftmaxCrossEntropyLoss
    }
    
    if name.lower() not in loss_functions:
        available = ', '.join(loss_functions.keys())
        raise ValueError(f"Unknown loss function: {name}. Available: {available}")
    
    return loss_functions[name.lower()](**kwargs)