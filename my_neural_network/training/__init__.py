"""
Training Module - Contains training utilities
"""

from .trainer import Trainer
from .callbacks import EarlyStopping

# Define what gets imported with "from my_neural_network.training import *"
__all__ = [
    'Trainer', 'EarlyStopping'
]