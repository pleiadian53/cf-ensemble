"""
PyTorch Gradient Descent optimization for CF-Ensemble.

This module provides an alternative to ALS using PyTorch's automatic differentiation
and modern optimizers. Should converge to the same solution as ALS for the basic
reconstruction objective.

Key advantages over ALS:
- GPU acceleration
- Modern optimizers (Adam, AdamW)
- Easy to extend with neural components
- Vectorized operations

Usage:
    >>> from cfensemble.optimization.pytorch_gd import PyTorchCFOptimizer
    >>> optimizer = PyTorchCFOptimizer(
    ...     m=10, n=100, latent_dim=20,
    ...     lambda_reg=0.01, device='cuda'
    ... )
    >>> optimizer.fit(R, C, max_iter=100)
    >>> X, Y = optimizer.get_factors()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Literal, Dict, List


class PyTorchCFOptimizer:
    """
    PyTorch-based optimizer for CF-Ensemble matrix factorization.
    
    Optimizes the same objective as ALS:
        L = Σ c_ui(r_ui - x_u^T y_i)² + λ(||X||² + ||Y||²)
    
    But uses gradient descent instead of closed-form updates.
    
    Parameters
    ----------
    m : int
        Number of classifiers
    n : int
        Number of instances
    latent_dim : int, default=20
        Dimensionality of latent factors
    lambda_reg : float, default=0.01
        L2 regularization parameter
    optimizer_type : {'adam', 'sgd', 'adamw'}, default='adam'
        Optimizer to use
    lr : float, default=0.01
        Learning rate
    device : {'cpu', 'cuda', 'mps'}, default='cpu'
        Device to run on
    
    Attributes
    ----------
    X : torch.Tensor, shape (d, m)
        Classifier latent factors
    Y : torch.Tensor, shape (d, n)
        Instance latent factors
    history : dict
        Training history (loss, reconstruction_loss, regularization_loss)
    
    Examples
    --------
    Basic usage:
    
    >>> optimizer = PyTorchCFOptimizer(m=10, n=100, latent_dim=20)
    >>> optimizer.fit(R, C, max_iter=50)
    >>> X, Y = optimizer.get_factors()
    
    With GPU:
    
    >>> optimizer = PyTorchCFOptimizer(
    ...     m=10, n=100, latent_dim=20,
    ...     device='cuda'
    ... )
    >>> optimizer.fit(R, C, max_iter=50)
    
    Compare with ALS:
    
    >>> from cfensemble.optimization import update_classifier_factors, update_instance_factors
    >>> 
    >>> # PyTorch
    >>> opt_pytorch = PyTorchCFOptimizer(m, n, d)
    >>> opt_pytorch.fit(R, C, max_iter=50)
    >>> X_torch, Y_torch = opt_pytorch.get_factors()
    >>> 
    >>> # ALS
    >>> X_als = np.random.randn(d, m) * 0.01
    >>> Y_als = np.random.randn(d, n) * 0.01
    >>> for _ in range(50):
    ...     X_als = update_classifier_factors(Y_als, R, C, lambda_reg)
    ...     Y_als = update_instance_factors(X_als, R, C, lambda_reg)
    >>> 
    >>> # Compare
    >>> print(f"Reconstruction error - PyTorch: {opt_pytorch.reconstruction_error():.4f}")
    >>> print(f"Reconstruction error - ALS: {compute_reconstruction_error(X_als, Y_als, R, C):.4f}")
    """
    
    def __init__(
        self,
        m: int,
        n: int,
        latent_dim: int = 20,
        lambda_reg: float = 0.01,
        optimizer_type: Literal['adam', 'sgd', 'adamw'] = 'adam',
        lr: float = 0.1,  # Higher default LR for matrix factorization
        device: Literal['cpu', 'cuda', 'mps'] = 'cpu',
        random_seed: Optional[int] = None
    ):
        """Initialize PyTorch CF optimizer."""
        self.m = m
        self.n = n
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.device = device
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Initialize latent factors as nn.Parameter for proper gradient tracking
        # Use torch.nn.Parameter to ensure they are leaf tensors
        self.X = nn.Parameter(torch.randn(latent_dim, m, device=device) * 0.01)
        self.Y = nn.Parameter(torch.randn(latent_dim, n, device=device) * 0.01)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'reconstruction_loss': [],
            'regularization_loss': []
        }
        
        self._is_fitted = False
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create PyTorch optimizer."""
        params = [self.X, self.Y]
        
        if self.optimizer_type == 'adam':
            return optim.Adam(params, lr=self.lr)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(params, lr=self.lr, momentum=0.9)
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def compute_loss(
        self,
        R: torch.Tensor,
        C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss, reconstruction loss, and regularization loss.
        
        Parameters
        ----------
        R : torch.Tensor, shape (m, n)
            Probability matrix
        C : torch.Tensor, shape (m, n)
            Confidence weights
        
        Returns
        -------
        total_loss : torch.Tensor
            Total loss (reconstruction + regularization)
        recon_loss : torch.Tensor
            Reconstruction loss only
        reg_loss : torch.Tensor
            Regularization loss only
        """
        # Reconstruction: X.T @ Y
        R_hat = self.X.T @ self.Y  # (m, n)
        
        # Weighted squared error
        residuals = R - R_hat
        recon_loss = torch.sum(C * residuals ** 2)
        
        # L2 regularization
        reg_loss = self.lambda_reg * (torch.sum(self.X ** 2) + torch.sum(self.Y ** 2))
        
        # Total loss
        total_loss = recon_loss + reg_loss
        
        return total_loss, recon_loss, reg_loss
    
    def fit(
        self,
        R: np.ndarray,
        C: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = True
    ) -> 'PyTorchCFOptimizer':
        """
        Fit the model using gradient descent.
        
        Parameters
        ----------
        R : np.ndarray, shape (m, n)
            Probability matrix
        C : np.ndarray, shape (m, n)
            Confidence weights
        max_iter : int, default=50
            Maximum number of iterations
        tol : float, default=1e-4
            Convergence tolerance (relative change in loss)
        verbose : bool, default=True
            Print progress
        
        Returns
        -------
        self : PyTorchCFOptimizer
            Fitted optimizer
        """
        # Convert to torch tensors
        R_torch = torch.from_numpy(R).float().to(self.device)
        C_torch = torch.from_numpy(C).float().to(self.device)
        
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            total_loss, recon_loss, reg_loss = self.compute_loss(R_torch, C_torch)
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Record history
            self.history['loss'].append(total_loss.item())
            self.history['reconstruction_loss'].append(recon_loss.item())
            self.history['regularization_loss'].append(reg_loss.item())
            
            # Check convergence
            relative_change = abs(prev_loss - total_loss.item()) / (abs(prev_loss) + 1e-10)
            
            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"Iter {iteration:3d}: Loss = {total_loss.item():10.4f} "
                      f"(Recon = {recon_loss.item():10.4f}, "
                      f"Reg = {reg_loss.item():8.4f}), "
                      f"Rel. Change = {relative_change:.2e}")
            
            # Convergence check
            if relative_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_loss = total_loss.item()
        else:
            if verbose:
                print(f"Max iterations ({max_iter}) reached without convergence")
        
        self._is_fitted = True
        return self
    
    def get_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent factors as NumPy arrays.
        
        Returns
        -------
        X : np.ndarray, shape (d, m)
            Classifier latent factors
        Y : np.ndarray, shape (d, n)
            Instance latent factors
        """
        return (
            self.X.detach().cpu().numpy(),
            self.Y.detach().cpu().numpy()
        )
    
    def reconstruction_error(self, R: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None) -> float:
        """
        Compute reconstruction error (without regularization).
        
        Parameters
        ----------
        R : np.ndarray, shape (m, n), optional
            Probability matrix. If None, uses the one from training.
        C : np.ndarray, shape (m, n), optional
            Confidence weights. If None, uses the one from training.
        
        Returns
        -------
        error : float
            Weighted reconstruction error
        """
        if not self._is_fitted and R is None:
            raise RuntimeError("Must fit model first or provide R")
        
        X_np, Y_np = self.get_factors()
        
        if R is None:
            # Use last computed reconstruction loss from history
            return self.history['reconstruction_loss'][-1]
        
        R_hat = X_np.T @ Y_np
        residuals = R - R_hat
        
        if C is None:
            C = np.ones_like(R)
        
        return np.sum(C * residuals ** 2)
    
    def predict(self) -> np.ndarray:
        """
        Predict the reconstructed probability matrix.
        
        Returns
        -------
        R_hat : np.ndarray, shape (m, n)
            Reconstructed probability matrix
        """
        X_np, Y_np = self.get_factors()
        return X_np.T @ Y_np


def compare_als_vs_pytorch(
    R: np.ndarray,
    C: np.ndarray,
    latent_dim: int = 20,
    lambda_reg: float = 0.01,
    max_iter: int = 50,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare ALS and PyTorch gradient descent on the same problem.
    
    Parameters
    ----------
    R : np.ndarray, shape (m, n)
        Probability matrix
    C : np.ndarray, shape (m, n)
        Confidence weights
    latent_dim : int, default=20
        Latent dimension
    lambda_reg : float, default=0.01
        Regularization parameter
    max_iter : int, default=50
        Maximum iterations
    random_seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print comparison results
    
    Returns
    -------
    results : dict
        Comparison results including:
        - X_als, Y_als: ALS factors
        - X_pytorch, Y_pytorch: PyTorch factors
        - recon_error_als: ALS reconstruction error
        - recon_error_pytorch: PyTorch reconstruction error
        - final_loss_als: ALS final loss
        - final_loss_pytorch: PyTorch final loss
        - factor_correlation: Correlation between factors
    
    Examples
    --------
    >>> R = np.random.rand(10, 100)
    >>> C = np.abs(R - 0.5)
    >>> results = compare_als_vs_pytorch(R, C)
    >>> print(f"ALS error: {results['recon_error_als']:.4f}")
    >>> print(f"PyTorch error: {results['recon_error_pytorch']:.4f}")
    """
    from cfensemble.optimization.als import (
        update_classifier_factors,
        update_instance_factors,
        compute_reconstruction_error
    )
    
    m, n = R.shape
    
    if verbose:
        print("="*60)
        print("Comparing ALS vs PyTorch Gradient Descent")
        print("="*60)
        print(f"Problem size: {m} classifiers × {n} instances")
        print(f"Latent dim: {latent_dim}")
        print(f"Regularization: λ = {lambda_reg}")
        print(f"Max iterations: {max_iter}")
        print()
    
    # === ALS ===
    if verbose:
        print("Running ALS...")
    
    np.random.seed(random_seed)
    X_als = np.random.randn(latent_dim, m) * 0.01
    Y_als = np.random.randn(latent_dim, n) * 0.01
    
    als_losses = []
    for i in range(max_iter):
        X_als = update_classifier_factors(Y_als, R, C, lambda_reg)
        Y_als = update_instance_factors(X_als, R, C, lambda_reg)
        
        recon_error = compute_reconstruction_error(X_als, Y_als, R, C)
        reg_loss = lambda_reg * (np.sum(X_als**2) + np.sum(Y_als**2))
        total_loss = recon_error + reg_loss
        als_losses.append(total_loss)
        
        if verbose and (i % 10 == 0 or i == max_iter - 1):
            print(f"Iter {i:3d}: Loss = {total_loss:10.4f}")
    
    recon_error_als = compute_reconstruction_error(X_als, Y_als, R, C)
    final_loss_als = als_losses[-1]
    
    # === PyTorch ===
    if verbose:
        print("\nRunning PyTorch...")
    
    optimizer = PyTorchCFOptimizer(
        m=m, n=n,
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        optimizer_type='adam',
        lr=0.5,  # High LR for faster convergence in matrix factorization
        device='cpu',
        random_seed=random_seed
    )
    optimizer.fit(R, C, max_iter=max_iter, verbose=verbose)
    
    X_pytorch, Y_pytorch = optimizer.get_factors()
    recon_error_pytorch = optimizer.reconstruction_error(R, C)
    final_loss_pytorch = optimizer.history['loss'][-1]
    
    # === Comparison ===
    if verbose:
        print("\n" + "="*60)
        print("Comparison Results")
        print("="*60)
        print(f"{'Metric':<30} {'ALS':>12} {'PyTorch':>12} {'Diff':>12}")
        print("-"*60)
        print(f"{'Final Loss':<30} {final_loss_als:12.4f} {final_loss_pytorch:12.4f} "
              f"{abs(final_loss_als - final_loss_pytorch):12.4f}")
        print(f"{'Reconstruction Error':<30} {recon_error_als:12.4f} {recon_error_pytorch:12.4f} "
              f"{abs(recon_error_als - recon_error_pytorch):12.4f}")
        
        # Factor correlation (sign-invariant)
        X_corr = abs(np.corrcoef(X_als.flatten(), X_pytorch.flatten())[0, 1])
        Y_corr = abs(np.corrcoef(Y_als.flatten(), Y_pytorch.flatten())[0, 1])
        print(f"{'Factor Correlation (X)':<30} {X_corr:12.4f}")
        print(f"{'Factor Correlation (Y)':<30} {Y_corr:12.4f}")
        
        print("\n✓ Both methods should converge to similar solutions!")
    
    return {
        'X_als': X_als,
        'Y_als': Y_als,
        'X_pytorch': X_pytorch,
        'Y_pytorch': Y_pytorch,
        'recon_error_als': recon_error_als,
        'recon_error_pytorch': recon_error_pytorch,
        'final_loss_als': final_loss_als,
        'final_loss_pytorch': final_loss_pytorch,
        'als_losses': als_losses,
        'pytorch_losses': optimizer.history['loss'],
        'X_correlation': abs(np.corrcoef(X_als.flatten(), X_pytorch.flatten())[0, 1]),
        'Y_correlation': abs(np.corrcoef(Y_als.flatten(), Y_pytorch.flatten())[0, 1])
    }
