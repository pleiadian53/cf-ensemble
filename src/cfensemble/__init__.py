"""
CF-Ensemble: Meta-learning via Latent-Factor-Based Collaborative Filtering

A framework for ensemble classification using collaborative filtering techniques
to transform and combine base model predictions.

This package implements a KD-inspired CF-Ensemble approach that combines:
- Matrix factorization (collaborative filtering) for reconstruction
- Supervised learning for label prediction
- Combined objective: L = ρ·L_recon + (1-ρ)·L_sup
"""

__version__ = "0.1.0"

# Core data structures
from .data import EnsembleData

# Confidence strategies
from .data import (
    CertaintyConfidence,
    LabelAwareConfidence,
    CalibrationConfidence,
    AdaptiveConfidence,
    get_confidence_strategy
)

# Loss functions
from .objectives import reconstruction_loss, supervised_loss, combined_loss

# Aggregators
from .ensemble import MeanAggregator, WeightedAggregator

# Optimization
from .optimization import (
    update_classifier_factors,
    update_instance_factors,
    CFEnsembleTrainer
)

# Models
from .models import ReliabilityWeightModel

__all__ = [
    # Data
    "EnsembleData",
    
    # Confidence
    "CertaintyConfidence",
    "LabelAwareConfidence",
    "CalibrationConfidence",
    "AdaptiveConfidence",
    "get_confidence_strategy",
    
    # Losses
    "reconstruction_loss",
    "supervised_loss",
    "combined_loss",
    
    # Aggregators
    "MeanAggregator",
    "WeightedAggregator",
    
    # Optimization
    "update_classifier_factors",
    "update_instance_factors",
    "CFEnsembleTrainer",
    
    # Models
    "ReliabilityWeightModel",
]
