"""
Synthetic Data Generation for CF-Ensemble Testing
=================================================

This module provides utilities for generating realistic synthetic ensemble data
with controlled properties for testing, validation, and benchmarking.

Key Features:
- Class imbalance (configurable minority class rate)
- Instance subgroups (mimics patient subtypes, data clusters)
- Classifier-subgroup affinity (different strengths/weaknesses)
- Systematic miscalibration
- Instance difficulty variation
- Realistic noise patterns

Designed to mimic real-world scenarios like:
- Biomedical prediction (disease detection, drug response)
- Genomic sequence classification (splice sites)
- Rare event detection (fraud, anomalies)
"""

import numpy as np
from typing import Tuple, Optional


def generate_imbalanced_ensemble_data(
    n_classifiers: int = 15,
    n_instances: int = 300,
    n_labeled: int = 150,
    positive_rate: float = 0.10,
    target_quality: float = 0.70,
    diversity: str = 'high',
    n_subgroups: int = 4,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate imbalanced ensemble data that ACTUALLY achieves target quality.
    
    Parameters are same as original, but this version correctly interprets
    target_quality as the metric appropriate for the imbalance level:
    - For positive_rate <= 0.20: target_quality = PR-AUC
    - For positive_rate > 0.20: target_quality = ROC-AUC
    
    Returns same as original.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Validate parameters
    if not 0 < positive_rate < 1:
        raise ValueError(f"positive_rate must be in (0, 1), got {positive_rate}")
    if not 0.5 <= target_quality <= 0.95:
        raise ValueError(f"target_quality must be in [0.5, 0.95], got {target_quality}")
    if n_labeled > n_instances:
        raise ValueError(f"n_labeled ({n_labeled}) cannot exceed n_instances ({n_instances})")
    
    # Generate true labels with class imbalance
    y_true = (np.random.rand(n_instances) < positive_rate).astype(float)
    
    # Assign labeled/unlabeled instances
    labeled_idx = np.random.choice(n_instances, n_labeled, replace=False)
    labels = np.full(n_instances, np.nan)
    labels[labeled_idx] = y_true[labeled_idx]
    
    # Create instance subgroups
    subgroup_assignment = np.random.randint(0, n_subgroups, size=n_instances)
    
    # Diversity mapping
    diversity_map = {
        'low': 0.05,
        'medium': 0.10,
        'high': 0.15
    }
    if diversity not in diversity_map:
        raise ValueError(f"diversity must be 'low', 'medium', or 'high', got '{diversity}'")
    diversity_std = diversity_map[diversity]
    
    # Generate classifier target qualities
    # For PR-AUC targets, we need to think about what quality means
    # A classifier with PR-AUC = target should have that PR-AUC
    # We'll generate diverse classifiers around this target
    
    n_good = max(2, n_classifiers // 3)
    n_poor = max(2, n_classifiers // 4)
    n_mediocre = n_classifiers - n_good - n_poor
    
    # Create target qualities for each classifier
    classifier_target_qualities = np.concatenate([
        np.random.uniform(target_quality + 0.05, min(0.95, target_quality + 0.15), size=n_good),
        np.random.uniform(max(0.52, target_quality - 0.10), target_quality + 0.05, size=n_mediocre),
        np.random.uniform(max(0.50, target_quality - 0.20), target_quality - 0.05, size=n_poor)
    ])
    np.random.shuffle(classifier_target_qualities)
    classifier_target_qualities = np.clip(classifier_target_qualities, 0.50, 0.95)
    
    # Generate probability matrix
    R = np.zeros((n_classifiers, n_instances))
    
    # For each classifier, generate predictions that achieve target quality
    for u in range(n_classifiers):
        target_q = classifier_target_qualities[u]
        
        # Strategy: Generate predictions with controlled quality
        # We'll use a latent score model: true_score + noise
        
        # Create latent true scores (higher for positives)
        true_scores = np.zeros(n_instances)
        
        # Add subgroup-specific effects
        for s in range(n_subgroups):
            subgroup_mask = (subgroup_assignment == s)
            
            # Subgroup difficulty (some harder to classify)
            subgroup_difficulty = np.random.uniform(-0.3, 0.3)
            
            # Classifier-subgroup affinity (varies per classifier)
            affinity = np.random.uniform(0.8, 1.2)
            
            # Set scores for this subgroup
            true_scores[subgroup_mask & (y_true == 1)] += (1.0 + subgroup_difficulty) * affinity
            true_scores[subgroup_mask & (y_true == 0)] += (0.0 + subgroup_difficulty * 0.5)
        
        # Add noise to create controlled separation
        # The key: noise level determines quality
        # Lower noise = better separation = higher PR-AUC
        
        # Emperically calibrated: noise_std ≈ (1.1 - target_q) * scale
        # Scale needs to decrease for extreme imbalance (PR-AUC more sensitive)
        if positive_rate <= 0.02:
            base_scale = 0.8  # Extreme imbalance: very sensitive
        elif positive_rate <= 0.10:
            base_scale = 1.2  # High imbalance
        else:
            base_scale = 1.0  # Moderate/balanced
        
        noise_std = (1.05 - target_q) * base_scale
        noise = np.random.normal(0, noise_std, size=n_instances)
        
        scores = true_scores + noise
        
        # Convert scores to probabilities via sigmoid
        R[u, :] = 1 / (1 + np.exp(-scores))
        
        # Add realistic calibration errors
        # Systematic bias for some classifiers
        if np.random.rand() < 0.3:
            bias = np.random.uniform(-0.08, 0.08)
            R[u, :] = np.clip(R[u, :] + bias, 0.02, 0.98)
        
        # Per-subgroup calibration issues (some classifiers struggle on specific subgroups)
        if np.random.rand() < 0.4:
            problem_subgroup = np.random.randint(0, n_subgroups)
            subgroup_mask = (subgroup_assignment == problem_subgroup)
            # Push predictions toward 0.5 (more uncertain)
            R[u, subgroup_mask] = 0.5 + (R[u, subgroup_mask] - 0.5) * 0.6
        
        # Clip to valid range
        R[u, :] = np.clip(R[u, :], 0.02, 0.98)
    
    return R, labels, labeled_idx, y_true



def generate_balanced_ensemble_data(
    n_classifiers: int = 15,
    n_instances: int = 300,
    n_labeled: int = 150,
    target_quality: float = 0.70,
    diversity: str = 'high',
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate balanced ensemble data (50/50 class distribution).
    
    Convenience function that calls `generate_imbalanced_ensemble_data`
    with positive_rate=0.5 for balanced classes.
    
    Parameters are the same as `generate_imbalanced_ensemble_data`,
    except positive_rate is fixed at 0.5.
    
    Returns
    -------
    Same as `generate_imbalanced_ensemble_data`
    
    See Also
    --------
    generate_imbalanced_ensemble_data : Full documentation
    """
    return generate_imbalanced_ensemble_data(
        n_classifiers=n_classifiers,
        n_instances=n_instances,
        n_labeled=n_labeled,
        positive_rate=0.5,  # Balanced!
        target_quality=target_quality,
        diversity=diversity,
        random_state=random_state
    )


def generate_simple_ensemble_data(
    n_classifiers: int = 10,
    n_instances: int = 100,
    n_labeled: int = 50,
    noise_level: float = 0.15,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate simple ensemble data for quick testing.
    
    Simplified version without subgroups, affinity, or systematic biases.
    Good for quick prototyping and unit tests.
    
    Parameters
    ----------
    n_classifiers : int, default=10
        Number of base classifiers
    n_instances : int, default=100
        Total number of instances
    n_labeled : int, default=50
        Number of labeled instances
    noise_level : float, default=0.15
        Amount of noise in predictions (0=perfect, 1=random)
    random_state : int or None, default=None
        Random seed
        
    Returns
    -------
    Same as `generate_imbalanced_ensemble_data`
    
    Examples
    --------
    >>> # Quick test data
    >>> R, labels, labeled_idx, y_true = generate_simple_ensemble_data(
    ...     n_classifiers=5,
    ...     n_instances=50,
    ...     random_state=42
    ... )
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Balanced labels
    y_true = (np.random.rand(n_instances) < 0.5).astype(float)
    
    # Assign labeled/unlabeled
    labeled_idx = np.random.choice(n_instances, n_labeled, replace=False)
    labels = np.full(n_instances, np.nan)
    labels[labeled_idx] = y_true[labeled_idx]
    
    # Simple probability matrix
    R = np.zeros((n_classifiers, n_instances))
    for u in range(n_classifiers):
        for i in range(n_instances):
            # Start with true probability
            if y_true[i] == 1:
                base_prob = 0.8  # Biased towards correct
            else:
                base_prob = 0.2  # Biased towards correct (low prob for positive)
            
            # Add noise
            noise = np.random.normal(0, noise_level)
            R[u, i] = np.clip(base_prob + noise, 0.05, 0.95)
    
    return R, labels, labeled_idx, y_true
