"""
CF-Ensemble Optimization

This module contains optimization algorithms for CF-Ensemble.

Two approaches are provided:
1. ALS-based (trainer.py): Alternating Least Squares with gradient descent for aggregator
   - Fast per iteration (closed-form updates)
   - May have convergence issues due to alternating objectives
   
2. PyTorch-based (trainer_pytorch.py): Joint gradient descent for all parameters
   - Unified objective, guaranteed convergence
   - GPU-accelerated, modern optimizers (Adam, AdamW)
   - Recommended for production use
"""

from .als import update_classifier_factors, update_instance_factors, compute_reconstruction_error
from .trainer import CFEnsembleTrainer

# PyTorch implementation (optional dependency)
try:
    from .trainer_pytorch import CFEnsemblePyTorchTrainer, CFEnsembleNet
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    CFEnsemblePyTorchTrainer = None
    CFEnsembleNet = None

__all__ = [
    # ALS
    'update_classifier_factors',
    'update_instance_factors',
    'compute_reconstruction_error',
    # Trainers
    'CFEnsembleTrainer',
    'CFEnsemblePyTorchTrainer',
    'CFEnsembleNet',
    'PYTORCH_AVAILABLE',
]
