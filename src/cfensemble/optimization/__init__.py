"""
CF-Ensemble Optimization

This module contains optimization algorithms for CF-Ensemble.

Two approaches are provided:
1. ALS (Alternating Least Squares): Closed-form, CPU-based, stable
2. PyTorch Gradient Descent: GPU-accelerated, flexible, extensible
"""

from .als import update_classifier_factors, update_instance_factors, compute_reconstruction_error
from .trainer import CFEnsembleTrainer

# PyTorch implementation (optional dependency)
try:
    from .pytorch_gd import PyTorchCFOptimizer, compare_als_vs_pytorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchCFOptimizer = None
    compare_als_vs_pytorch = None

__all__ = [
    # ALS
    'update_classifier_factors',
    'update_instance_factors',
    'compute_reconstruction_error',
    # Trainer
    'CFEnsembleTrainer',
    # PyTorch (if available)
    'PyTorchCFOptimizer',
    'compare_als_vs_pytorch',
    'PYTORCH_AVAILABLE',
]
