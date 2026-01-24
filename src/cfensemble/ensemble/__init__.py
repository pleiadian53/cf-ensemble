"""
CF-Ensemble Integration Methods

This module contains aggregation functions for combining base model predictions.
"""

from .aggregators import BaseAggregator, MeanAggregator, WeightedAggregator

__all__ = ["BaseAggregator", "MeanAggregator", "WeightedAggregator"]
