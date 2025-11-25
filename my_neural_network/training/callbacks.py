import numpy as np
from typing import Optional, Dict



class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights
        """
        if patience <= 0:
            raise ValueError(f"patience must be positive. Got: {patience}")
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max'. Got: {mode}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score: float, model_state: Optional[Dict] = None) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            if model_state is not None and self.restore_best_weights:
                self.best_weights = model_state.copy()
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model_state is not None and self.restore_best_weights:
                self.best_weights = model_state.copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop
    
    def get_best_weights(self):
        """Get the best model weights."""
        return self.best_weights