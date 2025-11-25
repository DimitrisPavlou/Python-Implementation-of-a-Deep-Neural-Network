import numpy as np
from typing import Dict


class Metrics:
    """Collection of evaluation metrics."""
    
    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute accuracy."""
        if predictions.ndim == 2:
            predictions = np.argmax(predictions, axis=1)
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        return np.mean(predictions == targets)
    
    @staticmethod
    def top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
        """Compute top-k accuracy."""
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        
        top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
        correct = np.any(top_k_preds == targets.reshape(-1, 1), axis=1)
        return np.mean(correct)
    
    @staticmethod
    def precision_recall_f1(predictions: np.ndarray, targets: np.ndarray, 
                           average: str = 'macro') -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        if predictions.ndim == 2:
            predictions = np.argmax(predictions, axis=1)
        if targets.ndim == 2:
            targets = np.argmax(targets, axis=1)
        
        classes = np.unique(np.concatenate([predictions, targets]))
        
        if average == 'macro':
            precisions, recalls, f1s = [], [], []
            
            for cls in classes:
                tp = np.sum((predictions == cls) & (targets == cls))
                fp = np.sum((predictions == cls) & (targets != cls))
                fn = np.sum((predictions != cls) & (targets == cls))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            
            return {
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'f1': np.mean(f1s)
            }
        
        else:
            raise ValueError(f"Unsupported average type: {average}")