"""
Confidence weighting strategies for CF-Ensemble.

This module provides different strategies for computing confidence weights
for the probability matrix R. Higher confidence means the model trusts that
prediction more during optimization.

Usage:
    >>> from cfensemble.data.confidence import CertaintyConfidence
    >>> strategy = CertaintyConfidence()
    >>> C = strategy.compute(R, labels)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class ConfidenceStrategy(ABC):
    """Base class for confidence computation strategies."""
    
    @abstractmethod
    def compute(self, R: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute confidence matrix C from probability matrix R.
        
        Parameters
        ----------
        R : np.ndarray, shape (m, n)
            Probability matrix from classifiers
        labels : np.ndarray, shape (n,), optional
            Ground truth labels with NaN for unlabeled instances
            
        Returns
        -------
        C : np.ndarray, shape (m, n)
            Confidence weights, typically in [0, 1]
        """
        pass


class UniformConfidence(ConfidenceStrategy):
    """
    Uniform confidence: All predictions equally trusted.
    
    Formula:
        c_ui = 1 for all (u, i)
    
    Use case:
        - Baseline comparison
        - When you have no prior knowledge about classifier reliability
    """
    
    def compute(self, R: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Return all-ones confidence matrix."""
        return np.ones_like(R)


class CertaintyConfidence(ConfidenceStrategy):
    """
    Certainty-based confidence: Trust predictions far from threshold.
    
    Formula:
        c_ui = |r_ui - 0.5|
    
    Intuition:
        - r_ui = 0.9 or 0.1 → confident (c_ui = 0.4)
        - r_ui = 0.5 → uncertain (c_ui = 0.0)
    
    Use case:
        - General-purpose confidence
        - Works without labels
        - Good default choice
    """
    
    def compute(self, R: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute distance from decision threshold."""
        return np.abs(R - 0.5)


class LabelAwareConfidence(ConfidenceStrategy):
    """
    Label-aware confidence: Reward correct predictions on labeled data.
    
    Formula:
        For labeled instance i:
            c_ui = r_ui        if y_i = 1
            c_ui = 1 - r_ui    if y_i = 0
        For unlabeled instance i:
            c_ui = |r_ui - 0.5|
    
    Intuition:
        - If y_i = 1 and r_ui = 0.9 → high confidence (0.9)
        - If y_i = 1 and r_ui = 0.1 → low confidence (0.1)
        - Directly measures alignment with ground truth
    
    Use case:
        - When you have labeled data
        - Emphasizes correct predictions
        - Good for semi-supervised learning
    
    Note:
        Only uses labels during training. For test data, falls back to
        certainty-based confidence.
    """
    
    def compute(self, R: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute label-aware confidence."""
        # Default: certainty-based for all
        C = np.abs(R - 0.5)
        
        # If labels provided, update labeled instances
        if labels is not None:
            labeled_idx = ~np.isnan(labels)
            
            for i in np.where(labeled_idx)[0]:
                if labels[i] == 1:
                    # Reward high predictions for positive labels
                    C[:, i] = R[:, i]
                else:
                    # Reward low predictions for negative labels
                    C[:, i] = 1 - R[:, i]
        
        return C


class CalibrationConfidence(ConfidenceStrategy):
    """
    Calibration-based confidence: Weight by classifier performance.
    
    Formula:
        For each classifier u:
            brier_u = mean((r_ui - y_i)^2) over labeled i
            calibration_u = 1 - brier_u
            c_ui = max(calibration_u, floor) for all i
    
    Intuition:
        - Good classifiers (low Brier score) get higher weight
        - Bad classifiers get downweighted
        - Per-classifier weight, not per-cell
    
    Use case:
        - When classifiers have varying quality
        - Want to identify and downweight poor performers
        - Requires labeled data for calibration
    
    Parameters
    ----------
    floor : float, default=0.1
        Minimum confidence value (prevents zero weights)
    """
    
    def __init__(self, floor: float = 0.1):
        """
        Initialize calibration confidence strategy.
        
        Parameters
        ----------
        floor : float, default=0.1
            Minimum confidence value to prevent zero weights
        """
        self.floor = floor
    
    def compute(self, R: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute calibration-based confidence."""
        # If no labels, fall back to uniform
        if labels is None or np.all(np.isnan(labels)):
            return np.ones_like(R)
        
        m, n = R.shape
        C = np.zeros_like(R)
        labeled_idx = ~np.isnan(labels)
        
        # Compute per-classifier Brier score
        for u in range(m):
            # Brier score on labeled data
            brier = np.mean((R[u, labeled_idx] - labels[labeled_idx])**2)
            calibration = 1 - brier
            
            # Apply to all instances (not just labeled)
            C[u, :] = calibration
        
        # Apply floor to prevent zero/negative weights
        return np.maximum(C, self.floor)


class AdaptiveConfidence(ConfidenceStrategy):
    """
    Adaptive confidence: Combines multiple strategies with learned weights.
    
    Formula:
        C = α * certainty + β * calibration + γ * agreement
    
    Features:
        - Combines certainty, calibration, and instance agreement
        - Weights can be learned via cross-validation
        - More robust than single strategy
    
    Parameters
    ----------
    alpha : float, default=0.5
        Weight for certainty-based confidence
    beta : float, default=0.3
        Weight for calibration-based confidence
    gamma : float, default=0.2
        Weight for agreement-based confidence (std across classifiers)
    
    Use case:
        - When you want best of multiple strategies
        - Can tune weights via validation
        - Good for production systems
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Initialize adaptive confidence strategy.
        
        Parameters
        ----------
        alpha : float, default=0.5
            Weight for certainty component
        beta : float, default=0.3
            Weight for calibration component
        gamma : float, default=0.2
            Weight for agreement component
        """
        if not np.isclose(alpha + beta + gamma, 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.certainty_strategy = CertaintyConfidence()
        self.calibration_strategy = CalibrationConfidence()
    
    def compute(self, R: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute adaptive confidence as weighted combination."""
        # 1. Certainty component
        C_certainty = self.certainty_strategy.compute(R, labels)
        
        # 2. Calibration component
        C_calibration = self.calibration_strategy.compute(R, labels)
        
        # 3. Agreement component: Lower std = higher confidence
        # Inverse of normalized std across classifiers
        R_std = np.std(R, axis=0)  # (n,)
        R_std_normalized = R_std / (np.max(R_std) + 1e-8)
        C_agreement = 1 - R_std_normalized  # High agreement = low std = high confidence
        C_agreement = np.broadcast_to(C_agreement, R.shape)  # (m, n)
        
        # Combine
        C = (self.alpha * C_certainty + 
             self.beta * C_calibration + 
             self.gamma * C_agreement)
        
        return C


# Convenience function
def get_confidence_strategy(name: str, **kwargs) -> ConfidenceStrategy:
    """
    Factory function for confidence strategies.
    
    Parameters
    ----------
    name : str
        Strategy name: 'uniform', 'certainty', 'label_aware', 
        'calibration', or 'adaptive'
    **kwargs
        Strategy-specific parameters
        
    Returns
    -------
    strategy : ConfidenceStrategy
        Configured confidence strategy
        
    Examples
    --------
    >>> strategy = get_confidence_strategy('certainty')
    >>> C = strategy.compute(R)
    
    >>> strategy = get_confidence_strategy('calibration', floor=0.05)
    >>> C = strategy.compute(R, labels)
    """
    strategies = {
        'uniform': UniformConfidence,
        'certainty': CertaintyConfidence,
        'label_aware': LabelAwareConfidence,
        'calibration': CalibrationConfidence,
        'adaptive': AdaptiveConfidence,
    }
    
    if name not in strategies:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(strategies.keys())}"
        )
    
    return strategies[name](**kwargs)
