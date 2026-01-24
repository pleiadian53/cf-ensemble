"""
CF-Ensemble Models

This module will contain advanced model implementations like:
- Reliability weight models (Phase 3)
- Instance-dependent aggregators
- Learned confidence models

Legacy models (cf, cf_models, knn_models) are in archive/ for reference.
New implementations will be added here as we progress through the roadmap.
"""

# Phase 3 models
from .reliability import ReliabilityWeightModel

__all__ = ["ReliabilityWeightModel"]
