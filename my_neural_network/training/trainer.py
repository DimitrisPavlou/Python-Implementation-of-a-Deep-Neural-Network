import numpy as np
from typing import List, Optional, Dict, Callable
import time
import warnings
from collections import defaultdict
from ..utils.metrics import Metrics
from ..data.dataloader import DataLoader
from .callbacks import EarlyStopping 
from ..optim.optimizers import LRScheduler


class Trainer:
    """Modern trainer with comprehensive features."""
    
    def __init__(self, model: List, loss_fn, optimizer, 
                 metrics: Optional[Dict[str, Callable]] = None,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: List of layers (network)
            loss_fn: Loss function
            optimizer: Optimizer
            metrics: Dictionary of metric functions
            device: Device (for future GPU support)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or {'accuracy': Metrics.accuracy}
        self.device = device
        
        self.history = defaultdict(list)
        self.epoch = 0
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        output = X
        for layer in self.model:
            if hasattr(layer, 'forward'):
                output = layer.forward(output)
        return output
    
    def _backward_pass(self, grad_output: np.ndarray) -> None:
        """Backward pass through the model."""
        grad = grad_output
        for layer in reversed(self.model):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
    
    def _set_mode(self, training: bool) -> None:
        """Set model to training or evaluation mode."""
        for layer in self.model:
            if hasattr(layer, 'train') and hasattr(layer, 'eval'):
                if training:
                    layer.train()
                else:
                    layer.eval()
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        for name, metric_fn in self.metrics.items():
            try:
                results[name] = metric_fn(predictions, targets)
            except Exception as e:
                warnings.warn(f"Failed to compute metric {name}: {e}")
                results[name] = 0.0
        return results
    
    def train_epoch(self, train_loader: DataLoader, 
                   val_loader: Optional[DataLoader] = None,
                   verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        self._set_mode(True)
        
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        n_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Forward pass
            predictions = self._forward_pass(X_batch)
            
            # Compute loss
            if hasattr(self.loss_fn, 'forward'):
                loss = self.loss_fn.forward(predictions, y_batch)
                grad_output = self.loss_fn.backward(predictions, y_batch)
            else:
                # Fallback for simple loss functions
                loss = self.loss_fn(predictions, y_batch)
                grad_output = predictions - y_batch  # Simplified
            
            # Backward pass
            self.optimizer.pre_update()
            self._backward_pass(grad_output)
            
            # Update parameters
            for layer in self.model:
                if hasattr(layer, 'parameters'):
                    self.optimizer.update_params(layer)
            
            self.optimizer.post_update()
            
            # Accumulate metrics
            epoch_loss += loss
            batch_metrics = self._compute_metrics(predictions, y_batch)
            for name, value in batch_metrics.items():
                epoch_metrics[name] += value
            
            n_batches += 1
            
            if verbose and (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Loss: {loss:.6f} - "
                      f"Acc: {batch_metrics.get('accuracy', 0):.4f}")
        
        # Average metrics
        epoch_loss /= n_batches
        for name in epoch_metrics:
            epoch_metrics[name] /= n_batches
        
        # Validation
        val_results = {}
        if val_loader is not None:
            val_results = self.evaluate(val_loader, verbose=False)
        
        # Store history
        self.history['train_loss'].append(epoch_loss)
        for name, value in epoch_metrics.items():
            self.history[f'train_{name}'].append(value)
        
        if val_results:
            for name, value in val_results.items():
                self.history[f'val_{name}'].append(value)
        
        epoch_time = time.time() - start_time
        
        if verbose:
            log_str = f"Epoch {self.epoch + 1} ({epoch_time:.2f}s) - "
            log_str += f"Loss: {epoch_loss:.6f}"
            for name, value in epoch_metrics.items():
                log_str += f" - {name.capitalize()}: {value:.4f}"
            
            if val_results:
                for name, value in val_results.items():
                    log_str += f" - Val {name.capitalize()}: {value:.4f}"
            
            log_str += f" - LR: {self.optimizer.current_lr:.6f}"
            print(log_str)
        
        results = {'loss': epoch_loss, **epoch_metrics}
        if val_results:
            results.update({f'val_{k}': v for k, v in val_results.items()})
        
        return results
    
    def evaluate(self, data_loader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """Evaluate the model."""
        self._set_mode(False)
        
        total_loss = 0.0
        total_metrics = defaultdict(float)
        n_batches = 0
        n_samples = 0
        
        for X_batch, y_batch in data_loader:
            # Forward pass
            predictions = self._forward_pass(X_batch)
            
            # Compute loss
            if hasattr(self.loss_fn, 'forward'):
                loss = self.loss_fn.forward(predictions, y_batch)
            else:
                loss = self.loss_fn(predictions, y_batch)
            
            # Compute metrics
            batch_metrics = self._compute_metrics(predictions, y_batch)
            
            # Accumulate
            batch_size = len(X_batch)
            total_loss += loss * batch_size
            for name, value in batch_metrics.items():
                total_metrics[name] += value * batch_size
            
            n_samples += batch_size
            n_batches += 1
        
        # Average
        avg_loss = total_loss / n_samples
        avg_metrics = {name: value / n_samples for name, value in total_metrics.items()}
        
        if verbose:
            log_str = f"Evaluation - Loss: {avg_loss:.6f}"
            for name, value in avg_metrics.items():
                log_str += f" - {name.capitalize()}: {value:.4f}"
            print(log_str)
        
        return {'loss': avg_loss, **avg_metrics}
    
    def fit(self, train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            early_stopping: Optional[EarlyStopping] = None,
            lr_scheduler: Optional[LRScheduler] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""
        
        if verbose:
            print(f"Starting training for {epochs} epochs...")
            print(f"Train batches: {len(train_loader)}")
            if val_loader:
                print(f"Validation batches: {len(val_loader)}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_results = self.train_epoch(train_loader, val_loader, verbose)
            
            # Learning rate scheduling
            if lr_scheduler:
                if lr_scheduler.mode == 'plateau' and val_loader:
                    lr_scheduler.step(epoch, epoch_results.get('val_loss'))
                else:
                    lr_scheduler.step(epoch)
            
            # Early stopping
            if early_stopping and val_loader:
                val_loss = epoch_results.get('val_loss')
                if val_loss and early_stopping(val_loss):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        if verbose:
            print("Training completed!")
        
        return dict(self.history)