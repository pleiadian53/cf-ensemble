"""
Loss Functions for CF-Ensemble

This module implements the core loss functions:
1. Reconstruction loss: Weighted matrix factorization
2. Supervised loss: Cross-entropy on aggregated predictions
3. Combined loss: The KD-inspired objective combining both

Based on: docs/methods/cf_ensemble_optimization_objective_tutorial.md
"""

import numpy as np
from typing import Dict, Any


def reconstruction_loss(
    R: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    lambda_reg: float
) -> float:
    """
    Reconstruction loss with regularization.
    
    L_recon = Σ c_ui(r_ui - x_u^T y_i)² + λ(||X||² + ||Y||²)
    
    This term encourages the latent factors to faithfully reconstruct
    the probability matrix, weighted by confidence.
    
    Parameters:
        R: Probability matrix (m × n)
        X: Classifier latent factors (d × m)
        Y: Instance latent factors (d × n)
        C: Confidence weights (m × n)
        lambda_reg: Regularization strength
    
    Returns:
        loss: Scalar reconstruction loss
    
    Example:
        >>> m, n, d = 10, 1000, 20
        >>> R = np.random.rand(m, n)
        >>> X = np.random.randn(d, m) * 0.01
        >>> Y = np.random.randn(d, n) * 0.01
        >>> C = np.ones((m, n))
        >>> loss = reconstruction_loss(R, X, Y, C, lambda_reg=0.01)
    """
    # Reconstruct probability matrix
    R_hat = X.T @ Y  # (m × n)
    
    # Weighted squared error
    residuals = R - R_hat
    weighted_mse = np.sum(C * residuals**2)
    
    # Regularization term (Frobenius norm squared)
    reg_term = lambda_reg * (np.sum(X**2) + np.sum(Y**2))
    
    return weighted_mse + reg_term


def supervised_loss(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    labeled_idx: np.ndarray,
    aggregator: Any
) -> float:
    """
    Supervised loss on labeled instances.
    
    L_sup = Σ_{i ∈ L} CE(y_i, g(r̂_·i))
    
    This term ensures the aggregated predictions match ground truth labels.
    
    Parameters:
        X: Classifier latent factors (d × m)
        Y: Instance latent factors (d × n)
        labels: Ground truth labels (n,) with NaN for unlabeled
        labeled_idx: Boolean mask for labeled instances
        aggregator: Aggregator object with predict() method
    
    Returns:
        loss: Scalar supervised loss (binary cross-entropy)
    
    Example:
        >>> from cfensemble.ensemble import MeanAggregator
        >>> X = np.random.randn(20, 10) * 0.01
        >>> Y = np.random.randn(20, 1000) * 0.01
        >>> labels = np.random.randint(0, 2, 1000).astype(float)
        >>> labels[500:] = np.nan
        >>> labeled_idx = ~np.isnan(labels)
        >>> agg = MeanAggregator()
        >>> loss = supervised_loss(X, Y, labels, labeled_idx, agg)
    """
    # Reconstruct probabilities for labeled instances
    R_hat = X.T @ Y  # (m × n)
    R_hat_labeled = R_hat[:, labeled_idx]  # (m × n_labeled)
    
    # Get aggregated predictions
    y_pred = aggregator.predict(R_hat_labeled)  # (n_labeled,)
    y_true = labels[labeled_idx]  # (n_labeled,)
    
    # Binary cross-entropy with numerical stability
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    
    ce = -y_true * np.log(y_pred_clipped) - (1 - y_true) * np.log(1 - y_pred_clipped)
    
    return np.mean(ce)


def combined_loss(
    R: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    labels: np.ndarray,
    labeled_idx: np.ndarray,
    aggregator: Any,
    rho: float,
    lambda_reg: float
) -> tuple[float, Dict[str, float]]:
    """
    Combined KD-inspired objective.
    
    L = ρ·L_recon + (1-ρ)·L_sup
    
    This is the core innovation: combining matrix reconstruction (soft targets)
    with supervised learning (hard labels), inspired by knowledge distillation.
    
    Parameters:
        R: Probability matrix (m × n)
        X: Classifier latent factors (d × m)
        Y: Instance latent factors (d × n)
        C: Confidence weights (m × n)
        labels: Ground truth labels (n,)
        labeled_idx: Boolean mask for labeled instances
        aggregator: Aggregator object
        rho: Trade-off parameter ∈ [0, 1]
            - rho=1: Pure reconstruction (old approach, reproduces errors)
            - rho=0: Pure supervised (ignores probability structure)
            - rho∈[0.3,0.7]: Recommended range
        lambda_reg: Regularization strength
    
    Returns:
        total_loss: Scalar combined loss
        loss_dict: Dictionary with component losses for logging
            - 'total': Total loss
            - 'reconstruction': Reconstruction component
            - 'supervised': Supervised component
            - 'recon_weighted': rho * reconstruction
            - 'sup_weighted': (1-rho) * supervised
            - 'rho': Trade-off parameter value
    
    Example:
        >>> from cfensemble.ensemble import WeightedAggregator
        >>> m, n, d = 10, 1000, 20
        >>> R = np.random.rand(m, n)
        >>> X = np.random.randn(d, m) * 0.01
        >>> Y = np.random.randn(d, n) * 0.01
        >>> C = np.ones((m, n))
        >>> labels = np.random.randint(0, 2, 1000).astype(float)
        >>> labels[500:] = np.nan
        >>> labeled_idx = ~np.isnan(labels)
        >>> agg = WeightedAggregator(m)
        >>> loss, loss_dict = combined_loss(
        ...     R, X, Y, C, labels, labeled_idx, agg,
        ...     rho=0.5, lambda_reg=0.01
        ... )
        >>> print(f"Total: {loss:.4f}, Recon: {loss_dict['reconstruction']:.4f}")
    """
    # Compute component losses
    l_recon = reconstruction_loss(R, X, Y, C, lambda_reg)
    l_sup = supervised_loss(X, Y, labels, labeled_idx, aggregator)
    
    # Weighted combination
    total = rho * l_recon + (1 - rho) * l_sup
    
    # Return total and components for logging/monitoring
    loss_dict = {
        'total': total,
        'reconstruction': l_recon,
        'supervised': l_sup,
        'recon_weighted': rho * l_recon,
        'sup_weighted': (1 - rho) * l_sup,
        'rho': rho
    }
    
    return total, loss_dict


def compute_rmse(R: np.ndarray, R_hat: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute RMSE for reconstruction quality.
    
    Useful for monitoring how well factors reconstruct the probability matrix.
    
    Parameters:
        R: Original probability matrix (m × n)
        R_hat: Reconstructed probability matrix (m × n)
        mask: Optional boolean mask to compute RMSE on subset
    
    Returns:
        rmse: Root mean squared error
    """
    residuals = R - R_hat
    if mask is not None:
        residuals = residuals[mask]
    return np.sqrt(np.mean(residuals**2))
