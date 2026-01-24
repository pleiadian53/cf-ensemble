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
    Generate realistic imbalanced ensemble data with controlled properties.
    
    Creates synthetic probability matrix with realistic complexity:
    - Class imbalance (minority class positives)
    - Instance subgroups with varying difficulty
    - Classifier-specific strengths/weaknesses per subgroup
    - Systematic miscalibration
    - Classifiers that are "confidently wrong" on specific subgroups
    
    This mimics real-world scenarios where:
    - Data has natural clusters (e.g., patient subtypes, sequence motifs)
    - Classifiers have different expertise (e.g., algorithm A excels on subgroup X)
    - Predictions are noisy and miscalibrated
    - Some classifiers fail systematically on certain data types
    
    Parameters
    ----------
    n_classifiers : int, default=15
        Number of base classifiers (m)
    n_instances : int, default=300
        Total number of instances (n)
    n_labeled : int, default=150
        Number of labeled instances (for semi-supervised setting)
    positive_rate : float, default=0.10
        Fraction of instances that are positive (minority class)
        Common values:
        - 0.10: Disease detection (10% prevalence)
        - 0.05: Rare disease (5% prevalence)
        - 0.01: Splice sites (1% of genomic positions)
        - 0.001: Very rare events
    target_quality : float, default=0.70
        Target average classifier quality
        - For balanced data: interpret as ROC-AUC
        - For imbalanced data: interpret as PR-AUC
        Range: [0.50, 0.95] (0.50 = random, 0.95 = near perfect)
    diversity : str, default='high'
        Classifier quality diversity level
        - 'low': Narrow quality range (std=0.03)
        - 'medium': Moderate quality range (std=0.08)
        - 'high': Wide quality range (std=0.12)
        High diversity means classifiers have very different strengths
    n_subgroups : int, default=4
        Number of instance subgroups (data clusters)
        Each subgroup has different difficulty and classifier affinity
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Returns
    -------
    R : np.ndarray, shape (m, n)
        Probability matrix where R[u, i] is classifier u's predicted
        probability that instance i belongs to the positive class
    labels : np.ndarray, shape (n,)
        Ground truth labels (0=negative, 1=positive) with NaN for unlabeled
    labeled_idx : np.ndarray, shape (n_labeled,)
        Indices of labeled instances
    y_true : np.ndarray, shape (n,)
        Complete ground truth labels (for evaluation only)
        
    Examples
    --------
    >>> # Generate data for disease detection (10% prevalence)
    >>> R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
    ...     n_classifiers=10,
    ...     n_instances=200,
    ...     positive_rate=0.10,
    ...     target_quality=0.65,
    ...     random_state=42
    ... )
    >>> R.shape
    (10, 200)
    >>> np.mean(y_true)  # Check actual positive rate
    ~0.10
    
    >>> # Generate very imbalanced data (splice sites: 1% positives)
    >>> R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
    ...     positive_rate=0.01,  # 99% negatives!
    ...     target_quality=0.50,  # Harder problem
    ...     diversity='high',
    ...     random_state=42
    ... )
    
    Notes
    -----
    - For balanced data (positive_rate ~0.5), use ROC-AUC to measure quality
    - For imbalanced data (positive_rate < 0.3), use PR-AUC to measure quality
    - The generated data is challenging: includes miscalibration, subgroup biases,
      and systematic failures that make ensemble learning non-trivial
    - Quality targets are approximate; actual achieved quality may vary due to
      complexity and randomness
      
    See Also
    --------
    generate_balanced_ensemble_data : For balanced class distributions
    generate_simple_ensemble_data : For quick testing without complexity
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
    
    # Create instance subgroups (mimics natural data clusters)
    subgroup_assignment = np.random.randint(0, n_subgroups, size=n_instances)
    
    # Each subgroup has different base difficulty (easy to hard)
    subgroup_difficulty = np.linspace(0.1, 0.7, n_subgroups)
    instance_difficulty = subgroup_difficulty[subgroup_assignment]
    
    # Diversity mapping - controls spread of classifier qualities
    diversity_map = {
        'low': 0.03,    # All classifiers similar
        'medium': 0.08,  # Moderate variation
        'high': 0.12    # Wide variation (some excellent, some poor)
    }
    if diversity not in diversity_map:
        raise ValueError(f"diversity must be 'low', 'medium', or 'high', got '{diversity}'")
    diversity_std = diversity_map[diversity]
    
    # Generate classifier qualities around target
    # Create diverse pool: good, mediocre, poor performers
    n_good = max(2, n_classifiers // 4)
    n_poor = max(2, n_classifiers // 4)
    n_mediocre = n_classifiers - n_good - n_poor
    
    # Scale qualities around target average
    quality_offset = target_quality - 0.70  # 0.70 is baseline
    classifier_qualities = np.concatenate([
        np.random.uniform(0.78 + quality_offset, 0.88 + quality_offset, size=n_good),
        np.random.uniform(0.65 + quality_offset, 0.78 + quality_offset, size=n_mediocre),
        np.random.uniform(0.55 + quality_offset, 0.68 + quality_offset, size=n_poor)
    ])
    np.random.shuffle(classifier_qualities)
    classifier_qualities = np.clip(classifier_qualities, 0.52, 0.95)
    
    # Classifier-subgroup affinity: Each classifier has different strengths per subgroup
    # Affinity > 1.0 means classifier excels on this subgroup
    # Affinity < 1.0 means classifier struggles on this subgroup
    classifier_subgroup_affinity = np.random.uniform(0.55, 1.45, size=(n_classifiers, n_subgroups))
    
    # Generate probability matrix
    R = np.zeros((n_classifiers, n_instances))
    
    # Adaptive noise level based on target quality
    base_noise = 0.23 if target_quality > 0.65 else 0.28
    
    # Generate predictions for each classifier
    for u in range(n_classifiers):
        base_quality = classifier_qualities[u]
        
        for i in range(n_instances):
            subgroup = subgroup_assignment[i]
            
            # Effective quality depends on:
            # 1. Classifier's base quality
            # 2. Affinity to this subgroup
            # 3. Instance difficulty
            affinity = classifier_subgroup_affinity[u, subgroup]
            effective_quality = base_quality * affinity * (1 - instance_difficulty[i] * 0.5)
            effective_quality = np.clip(effective_quality, 0.2, 0.95)
            
            # Generate prediction (correct with probability = effective_quality)
            if np.random.rand() < effective_quality:
                # Correct prediction
                base_prob = y_true[i]
                # Add miscalibration (varies by subgroup)
                bias = np.random.uniform(-0.2, 0.2)
                noise = np.random.normal(0, base_noise * (1 + instance_difficulty[i] * 0.5))
            else:
                # Incorrect prediction
                base_prob = 1 - y_true[i]
                # Wrong predictions more uncertain on hard instances
                bias = np.random.uniform(-0.25, 0.0)
                noise = np.random.normal(0, base_noise * (1 + instance_difficulty[i] * 0.8))
            
            # Apply bias and noise, clip to valid probability range
            R[u, i] = np.clip(base_prob + bias + noise, 0.05, 0.95)
    
    # Add systematic miscalibration (40% of classifiers have consistent biases)
    for u in range(n_classifiers):
        if np.random.rand() < 0.4:
            bias = np.random.uniform(-0.15, 0.15)
            R[u, :] = np.clip(R[u, :] + bias, 0.05, 0.95)
    
    # Add classifiers that are "confidently wrong" on specific subgroups
    # This mimics algorithmic biases and creates opportunities for reliability learning
    for u in range(n_classifiers // 3):
        target_subgroup = np.random.randint(0, n_subgroups)
        subgroup_mask = subgroup_assignment == target_subgroup
        
        if np.random.rand() < 0.6:  # 60% chance this classifier struggles here
            for i in np.where(subgroup_mask)[0]:
                if np.random.rand() < 0.3:
                    # Flip towards wrong answer
                    R[u, i] = 0.5 + (0.5 - R[u, i]) * 0.4
                else:
                    # Just reduce confidence
                    R[u, i] = 0.5 + (R[u, i] - 0.5) * 0.7
    
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
