"""
EnsembleData: Container for CF-Ensemble data structures.

This module provides the core data structure for holding:
- Probability matrix R from base models
- Ground truth labels (with NaN for unlabeled)
- Confidence/reliability weights
"""

import numpy as np
from typing import Optional, Tuple


class EnsembleData:
    """
    Container for CF-Ensemble data structures.
    
    Attributes:
        R (np.ndarray): Probability matrix (m × n) where m = # classifiers, n = # instances
        labels (np.ndarray): Ground truth labels (n,) with np.nan for unlabeled points
        C (np.ndarray): Confidence weights (m × n) indicating trust in each prediction
        labeled_idx (np.ndarray): Boolean mask for labeled instances
        unlabeled_idx (np.ndarray): Boolean mask for unlabeled instances
        n_classifiers (int): Number of base classifiers
        n_instances (int): Total number of instances (labeled + unlabeled)
        n_labeled (int): Number of labeled instances
        n_unlabeled (int): Number of unlabeled instances
    
    Example:
        >>> R = np.random.rand(10, 1000)  # 10 classifiers, 1000 instances
        >>> labels = np.random.randint(0, 2, 1000).astype(float)
        >>> labels[500:] = np.nan  # Mark last 500 as unlabeled
        >>> data = EnsembleData(R, labels)
        >>> print(f"Labeled: {data.n_labeled}, Unlabeled: {data.n_unlabeled}")
    """
    
    def __init__(
        self,
        R: np.ndarray,
        labels: np.ndarray,
        C: Optional[np.ndarray] = None
    ):
        """
        Initialize EnsembleData.
        
        Parameters:
            R: Probability matrix (m × n)
                - m = number of base classifiers
                - n = number of data points
                - R[u, i] = classifier u's predicted probability for point i
            labels: Ground truth labels (n,)
                - Use np.nan for unlabeled/test points
                - Binary: 0 or 1 for labeled points
            C: Confidence weights (m × n), optional
                - If None, computed automatically using certainty-based strategy
                - C[u, i] = trust/confidence in prediction R[u, i]
        
        Raises:
            ValueError: If dimensions don't match or data is invalid
        """
        # Validate inputs
        if R.ndim != 2:
            raise ValueError(f"R must be 2D array, got shape {R.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D array, got shape {labels.shape}")
        if R.shape[1] != len(labels):
            raise ValueError(
                f"R columns ({R.shape[1]}) must match labels length ({len(labels)})"
            )
        if not np.all((R >= 0) & (R <= 1)):
            raise ValueError("R must contain probabilities in [0, 1]")
        
        # Store data
        self.R = R
        self.labels = labels
        
        # Compute confidence if not provided
        if C is None:
            self.C = self._compute_confidence()
        else:
            if C.shape != R.shape:
                raise ValueError(f"C shape {C.shape} must match R shape {R.shape}")
            if not np.all(C >= 0):
                raise ValueError("C must contain non-negative weights")
            self.C = C
        
        # Create masks
        self.labeled_idx = ~np.isnan(labels)
        self.unlabeled_idx = np.isnan(labels)
        
        # Store dimensions
        self.n_classifiers, self.n_instances = R.shape
        self.n_labeled = np.sum(self.labeled_idx)
        self.n_unlabeled = np.sum(self.unlabeled_idx)
        
        # Check that labeled values are binary (if any exist)
        if self.n_labeled > 0:
            labeled_vals = labels[self.labeled_idx]
            if not np.all(np.isin(labeled_vals, [0, 1])):
                raise ValueError("Labeled instances must be 0 or 1")
    
    def _compute_confidence(self) -> np.ndarray:
        """
        Compute default confidence weights using certainty-based strategy.
        
        Strategy: c_ui = |r_ui - 0.5|
        - Predictions close to 0.5 have low confidence (uncertain)
        - Predictions close to 0 or 1 have high confidence (certain)
        
        Returns:
            C: Confidence matrix (m × n)
        """
        return np.abs(self.R - 0.5)
    
    def get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for labeled instances only.
        
        Returns:
            R_labeled: Probability matrix for labeled instances (m × n_labeled)
            y_labeled: Labels for labeled instances (n_labeled,)
            C_labeled: Confidence for labeled instances (m × n_labeled)
        """
        return (
            self.R[:, self.labeled_idx],
            self.labels[self.labeled_idx],
            self.C[:, self.labeled_idx]
        )
    
    def get_unlabeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for unlabeled instances only.
        
        Returns:
            R_unlabeled: Probability matrix for unlabeled instances (m × n_unlabeled)
            C_unlabeled: Confidence for unlabeled instances (m × n_unlabeled)
        """
        return (
            self.R[:, self.unlabeled_idx],
            self.C[:, self.unlabeled_idx]
        )
    
    def split_labeled_data(
        self,
        train_fraction: float = 0.8,
        random_state: Optional[int] = None
    ) -> Tuple['EnsembleData', 'EnsembleData']:
        """
        Split labeled data into train and validation sets.
        
        Useful for hyperparameter tuning. Preserves unlabeled data in both splits
        for transductive learning.
        
        Parameters:
            train_fraction: Fraction of labeled data for training
            random_state: Random seed for reproducibility
        
        Returns:
            train_data: EnsembleData with training split
            val_data: EnsembleData with validation split
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get labeled indices
        labeled_indices = np.where(self.labeled_idx)[0]
        n_train = int(len(labeled_indices) * train_fraction)
        
        # Shuffle and split
        shuffled = np.random.permutation(labeled_indices)
        train_indices = shuffled[:n_train]
        val_indices = shuffled[n_train:]
        
        # Create new label arrays
        train_labels = self.labels.copy()
        train_labels[val_indices] = np.nan  # Mask validation as unlabeled
        
        val_labels = self.labels.copy()
        val_labels[train_indices] = np.nan  # Mask train as unlabeled
        
        # Create new EnsembleData objects
        train_data = EnsembleData(self.R, train_labels, self.C)
        val_data = EnsembleData(self.R, val_labels, self.C)
        
        return train_data, val_data
    
    def update_confidence(self, C_new: np.ndarray):
        """
        Update confidence matrix (useful after learning reliability weights).
        
        Parameters:
            C_new: New confidence matrix (m × n)
        """
        if C_new.shape != self.R.shape:
            raise ValueError(f"C_new shape {C_new.shape} must match R shape {self.R.shape}")
        if not np.all(C_new >= 0):
            raise ValueError("C_new must contain non-negative weights")
        self.C = C_new
    
    def __repr__(self) -> str:
        return (
            f"EnsembleData("
            f"n_classifiers={self.n_classifiers}, "
            f"n_instances={self.n_instances}, "
            f"n_labeled={self.n_labeled}, "
            f"n_unlabeled={self.n_unlabeled})"
        )
    
    def summary(self) -> str:
        """Get detailed summary of the data."""
        lines = [
            "=" * 50,
            "EnsembleData Summary",
            "=" * 50,
            f"Number of classifiers:    {self.n_classifiers}",
            f"Total instances:          {self.n_instances}",
            f"  - Labeled:              {self.n_labeled} ({100*self.n_labeled/self.n_instances:.1f}%)",
            f"  - Unlabeled:            {self.n_unlabeled} ({100*self.n_unlabeled/self.n_instances:.1f}%)",
            "",
            "Probability matrix (R):",
            f"  - Shape:                {self.R.shape}",
            f"  - Range:                [{self.R.min():.3f}, {self.R.max():.3f}]",
            f"  - Mean:                 {self.R.mean():.3f}",
            "",
            "Confidence matrix (C):",
            f"  - Shape:                {self.C.shape}",
            f"  - Range:                [{self.C.min():.3f}, {self.C.max():.3f}]",
            f"  - Mean:                 {self.C.mean():.3f}",
            "",
            "Labels:",
            f"  - Positive class:       {np.sum(self.labels[self.labeled_idx] == 1)} ({100*np.mean(self.labels[self.labeled_idx]):.1f}%)",
            f"  - Negative class:       {np.sum(self.labels[self.labeled_idx] == 0)} ({100*(1-np.mean(self.labels[self.labeled_idx])):.1f}%)",
            "=" * 50,
        ]
        return "\n".join(lines)
