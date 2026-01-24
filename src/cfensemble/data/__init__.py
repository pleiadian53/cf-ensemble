"""
CF-Ensemble Data Structures

This module contains data handling and preprocessing utilities.
"""

from .ensemble_data import EnsembleData
from .confidence import (
    ConfidenceStrategy,
    UniformConfidence,
    CertaintyConfidence,
    LabelAwareConfidence,
    CalibrationConfidence,
    AdaptiveConfidence,
    get_confidence_strategy
)
from .synthetic import (
    generate_imbalanced_ensemble_data,
    generate_balanced_ensemble_data,
    generate_simple_ensemble_data
)

__all__ = [
    # Data structures
    "EnsembleData",
    # Confidence strategies
    "ConfidenceStrategy",
    "UniformConfidence",
    "CertaintyConfidence",
    "LabelAwareConfidence",
    "CalibrationConfidence",
    "AdaptiveConfidence",
    "get_confidence_strategy",
    # Synthetic data generation
    "generate_imbalanced_ensemble_data",
    "generate_balanced_ensemble_data",
    "generate_simple_ensemble_data"
]
