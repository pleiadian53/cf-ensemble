"""
Alternating Least Squares (ALS) update functions for CF-Ensemble.

This module implements closed-form updates for the latent factors X and Y
in the matrix factorization objective:

    L_recon = Σ c_ui(r_ui - x_u^T y_i)² + λ(||X||² + ||Y||²)

The ALS algorithm alternates between:
1. Fixing Y, updating X (classifier factors)
2. Fixing X, updating Y (instance factors)

Each update has a closed-form solution obtained by setting the gradient to zero.
"""

import numpy as np
from typing import Tuple


def update_classifier_factors(
    Y: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    lambda_reg: float
) -> np.ndarray:
    """
    Update classifier latent factors X given fixed instance factors Y.
    
    Uses the closed-form ALS solution:
    For each classifier u:
        x_u = (Y C_u Y^T + λI)^{-1} Y C_u r_u
    
    where:
    - C_u = diag(c_u1, c_u2, ..., c_un) is the confidence weights for classifier u
    - r_u is the u-th row of R (classifier u's predictions)
    
    Derivation:
        ∂L/∂x_u = -2 Σ_i c_ui(r_ui - x_u^T y_i)y_i + 2λx_u = 0
        => Σ_i c_ui y_i y_i^T x_u + λx_u = Σ_i c_ui r_ui y_i
        => (Y C_u Y^T + λI)x_u = Y C_u r_u
    
    Parameters:
        Y: Instance latent factors (d × n)
           Each column y_i is the latent representation of instance i
        R: Probability matrix (m × n)
           R[u,i] is classifier u's prediction probability for instance i
        C: Confidence weights (m × n)
           C[u,i] indicates confidence in prediction R[u,i]
        lambda_reg: Regularization parameter (λ > 0)
    
    Returns:
        X: Updated classifier latent factors (d × m)
           Each column x_u is the latent representation of classifier u
    
    Shape notation:
        d = latent dimension
        m = number of classifiers
        n = number of instances
    
    Example:
        >>> Y = np.random.randn(10, 1000)  # 10-dim latent, 1000 instances
        >>> R = np.random.rand(20, 1000)   # 20 classifiers
        >>> C = np.abs(R - 0.5)            # Certainty-based confidence
        >>> X = update_classifier_factors(Y, R, C, lambda_reg=0.01)
        >>> X.shape
        (10, 20)
    
    Notes:
        - Solves m independent d×d linear systems (one per classifier)
        - Uses np.linalg.solve for numerical stability
        - Can be vectorized for speed (see optimization notes)
        - O(m × d³) complexity per iteration
    """
    d, n = Y.shape
    m = R.shape[0]
    X = np.zeros((d, m))
    
    # Regularization term (constant across classifiers)
    lambda_I = lambda_reg * np.eye(d)
    
    for u in range(m):
        # Get confidence weights for this classifier
        c_u = C[u, :]  # (n,)
        
        # Weighted gram matrix: Y C_u Y^T
        # Y @ diag(c_u) @ Y.T = Y @ (c_u[:, None] * Y.T)
        Y_weighted = Y * c_u[None, :]  # Broadcasting: (d, n) * (1, n)
        A = Y_weighted @ Y.T + lambda_I  # (d, d)
        
        # Weighted target: Y C_u r_u = Y @ (c_u * r_u)
        r_u = R[u, :]  # (n,)
        b = Y_weighted @ r_u  # (d,)
        
        # Solve: A x_u = b
        X[:, u] = np.linalg.solve(A, b)
    
    return X


def update_instance_factors(
    X: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    lambda_reg: float
) -> np.ndarray:
    """
    Update instance latent factors Y given fixed classifier factors X.
    
    Uses the closed-form ALS solution:
    For each instance i:
        y_i = (X C_i X^T + λI)^{-1} X C_i r_i
    
    where:
    - C_i = diag(c_1i, c_2i, ..., c_mi) is the confidence weights for instance i
    - r_i is the i-th column of R (all classifier predictions for instance i)
    
    Derivation:
        ∂L/∂y_i = -2 Σ_u c_ui(r_ui - x_u^T y_i)x_u + 2λy_i = 0
        => Σ_u c_ui x_u x_u^T y_i + λy_i = Σ_u c_ui r_ui x_u
        => (X C_i X^T + λI)y_i = X C_i r_i
    
    Parameters:
        X: Classifier latent factors (d × m)
           Each column x_u is the latent representation of classifier u
        R: Probability matrix (m × n)
           R[u,i] is classifier u's prediction probability for instance i
        C: Confidence weights (m × n)
           C[u,i] indicates confidence in prediction R[u,i]
        lambda_reg: Regularization parameter (λ > 0)
    
    Returns:
        Y: Updated instance latent factors (d × n)
           Each column y_i is the latent representation of instance i
    
    Shape notation:
        d = latent dimension
        m = number of classifiers
        n = number of instances
    
    Example:
        >>> X = np.random.randn(10, 20)    # 10-dim latent, 20 classifiers
        >>> R = np.random.rand(20, 1000)   # 1000 instances
        >>> C = np.abs(R - 0.5)            # Certainty-based confidence
        >>> Y = update_instance_factors(X, R, C, lambda_reg=0.01)
        >>> Y.shape
        (10, 1000)
    
    Notes:
        - Solves n independent d×d linear systems (one per instance)
        - Uses np.linalg.solve for numerical stability
        - Can be vectorized for speed (see optimization notes)
        - O(n × d³) complexity per iteration
        - This step is typically the bottleneck (n >> m)
    """
    d, m = X.shape
    n = R.shape[1]
    Y = np.zeros((d, n))
    
    # Regularization term (constant across instances)
    lambda_I = lambda_reg * np.eye(d)
    
    for i in range(n):
        # Get confidence weights for this instance
        c_i = C[:, i]  # (m,)
        
        # Weighted gram matrix: X C_i X^T
        # X @ diag(c_i) @ X.T = X @ (c_i[:, None] * X.T)
        X_weighted = X * c_i[None, :]  # Broadcasting: (d, m) * (1, m)
        A = X_weighted @ X.T + lambda_I  # (d, d)
        
        # Weighted target: X C_i r_i = X @ (c_i * r_i)
        r_i = R[:, i]  # (m,)
        b = X_weighted @ r_i  # (d,)
        
        # Solve: A y_i = b
        Y[:, i] = np.linalg.solve(A, b)
    
    return Y


def compute_reconstruction_error(
    X: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    C: np.ndarray
) -> float:
    """
    Compute weighted reconstruction error (without regularization).
    
    Error = Σ_{u,i} c_ui (r_ui - x_u^T y_i)²
    
    This is useful for monitoring convergence separately from the regularization term.
    
    Parameters:
        X: Classifier factors (d × m)
        Y: Instance factors (d × n)
        R: Probability matrix (m × n)
        C: Confidence weights (m × n)
    
    Returns:
        error: Scalar reconstruction error
    
    Example:
        >>> error = compute_reconstruction_error(X, Y, R, C)
        >>> print(f"Reconstruction RMSE: {np.sqrt(error / (m*n)):.4f}")
    """
    R_hat = X.T @ Y  # (m × n)
    residuals = R - R_hat
    weighted_sq_error = np.sum(C * residuals ** 2)
    return weighted_sq_error


# Future optimization: Vectorized updates
# For large-scale problems, we can vectorize across classifiers/instances
# This requires solving multiple linear systems simultaneously using
# batched solvers or iterative methods like conjugate gradient.
#
# Potential speedup: 5-10x for m, n > 1000
#
# Example signature:
# def update_classifier_factors_vectorized(Y, R, C, lambda_reg):
#     # Solve all m systems simultaneously
#     pass
