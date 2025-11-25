
import numpy as np
from typing import Tuple, Optional, Iterator

# ==============================================================================
# DATA HANDLING
# ==============================================================================

class DataLoader:
    """DataLoader for batch handling with various sampling strategies."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                 shuffle: bool = True, drop_last: bool = False, seed: Optional[int] = None):
        """
        Initialize DataLoader.
        
        Args:
            X: Input features
            y: Target labels
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for reproducibility
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got: {batch_size}")
        
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.n_samples = len(X)
        self.n_batches = self.n_samples // batch_size
        if not drop_last and self.n_samples % batch_size != 0:
            self.n_batches += 1
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.n_samples)
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and end_idx - i < self.batch_size:
                break
            
            batch_indices = indices[i:end_idx]
            yield self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches
