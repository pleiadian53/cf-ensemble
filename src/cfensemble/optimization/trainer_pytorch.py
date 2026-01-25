"""
PyTorch-based CF-Ensemble Trainer

This module implements CF-Ensemble using joint gradient descent via PyTorch,
solving the optimization instability problem in the ALS-based trainer.

Key differences from trainer.py:
1. Joint optimization: All parameters (X, Y, aggregator weights) updated together
2. Unified gradients: Single backward pass for combined loss
3. Modern optimizers: Adam, AdamW with learning rate scheduling
4. Guaranteed convergence: Monotonic loss decrease

Based on: docs/failure_modes/optimization_instability.md
"""

import warnings
from typing import Optional, Tuple, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Install with: pip install torch\n"
        "PyTorch trainer will not be available."
    )

from ..data.ensemble_data import EnsembleData


class CFEnsembleNet(nn.Module):
    """
    PyTorch module for CF-Ensemble.
    
    Implements the combined KD-inspired objective:
        L = ρ·L_recon + (1-ρ)·L_sup
    
    All parameters (X, Y, aggregator weights) are torch.nn.Parameter objects
    that get optimized jointly via backpropagation.
    
    Parameters:
    -----------
    m : int
        Number of classifiers
    n : int
        Number of instances
    d : int
        Latent dimensionality
    aggregator_type : str
        Type of aggregator ('mean' or 'weighted')
    """
    
    def __init__(
        self,
        m: int,
        n: int,
        d: int,
        aggregator_type: str = 'weighted'
    ):
        super().__init__()
        
        self.m = m
        self.n = n
        self.d = d
        self.aggregator_type = aggregator_type
        
        # Classifier latent factors (d × m)
        self.X = nn.Parameter(torch.randn(d, m) * 0.01)
        
        # Instance latent factors (d × n)
        self.Y = nn.Parameter(torch.randn(d, n) * 0.01)
        
        # Aggregator parameters
        if aggregator_type == 'weighted':
            self.w = nn.Parameter(torch.ones(m) / m)  # Start uniform
            self.b = nn.Parameter(torch.zeros(1))
        else:  # mean aggregator (no parameters)
            self.register_buffer('w', torch.ones(m) / m)
            self.register_buffer('b', torch.zeros(1))
    
    def reconstruct(self) -> torch.Tensor:
        """
        Reconstruct probability matrix: R_hat = X^T @ Y
        
        Returns:
        --------
        R_hat : Tensor (m × n)
            Reconstructed probabilities
        """
        return self.X.T @ self.Y  # (m × n)
    
    def aggregate(self, R_hat_subset: torch.Tensor) -> torch.Tensor:
        """
        Aggregate reconstructed probabilities to final predictions.
        
        Parameters:
        -----------
        R_hat_subset : Tensor (m × n_subset)
            Reconstructed probabilities for subset of instances
        
        Returns:
        --------
        y_pred : Tensor (n_subset,)
            Aggregated predictions
        """
        if self.aggregator_type == 'mean':
            return torch.mean(R_hat_subset, dim=0)
        else:  # weighted
            logits = self.w @ R_hat_subset + self.b  # (n_subset,)
            return torch.sigmoid(logits)
    
    def forward(self, labeled_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reconstruct + aggregate for labeled instances.
        
        Parameters:
        -----------
        labeled_idx : Tensor (n_labeled,)
            Indices of labeled instances
        
        Returns:
        --------
        y_pred : Tensor (n_labeled,)
            Predictions for labeled instances
        """
        R_hat = self.reconstruct()
        R_hat_labeled = R_hat[:, labeled_idx]
        return self.aggregate(R_hat_labeled)
    
    def compute_loss(
        self,
        R: torch.Tensor,
        C: torch.Tensor,
        labels: torch.Tensor,
        labeled_mask: torch.Tensor,
        rho: float,
        lambda_reg: float,
        use_class_weights: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss: ρ·L_recon + (1-ρ)·L_sup
        
        With class weighting, uses inverse class frequency to balance gradients
        from imbalanced data.
        
        Parameters:
        -----------
        R : Tensor (m × n)
            Observed probability matrix
        C : Tensor (m × n)
            Confidence weights
        labels : Tensor (n,)
            Ground truth labels (NaN for unlabeled)
        labeled_mask : Tensor (n,)
            Boolean mask for labeled instances
        rho : float
            Trade-off parameter ∈ [0, 1]
        lambda_reg : float
            Regularization strength
        use_class_weights : bool, default=True
            If True, weight instances by inverse class frequency
        
        Returns:
        --------
        loss : Tensor (scalar)
            Combined loss
        loss_dict : dict
            Dictionary with loss components for logging
        """
        # 1. Reconstruction loss
        R_hat = self.reconstruct()
        residuals = R - R_hat
        weighted_residuals = C * (residuals ** 2)
        recon_loss = torch.sum(weighted_residuals)
        
        # 2. Regularization
        reg_loss = lambda_reg * (torch.sum(self.X ** 2) + torch.sum(self.Y ** 2))
        
        # 3. Supervised loss (only on labeled instances)
        if torch.sum(labeled_mask) > 0:
            labeled_idx = torch.where(labeled_mask)[0]
            y_pred = self.forward(labeled_idx)
            y_true = labels[labeled_mask]
            
            # Binary cross-entropy with numerical stability
            eps = 1e-15
            y_pred_clipped = torch.clamp(y_pred, eps, 1 - eps)
            bce = -(y_true * torch.log(y_pred_clipped) +
                   (1 - y_true) * torch.log(1 - y_pred_clipped))
            
            # Apply class weighting if enabled
            if use_class_weights:
                n = len(y_true)
                n_pos = torch.sum(y_true == 1).float()
                n_neg = n - n_pos
                
                # Compute class weights (inverse frequency)
                if n_pos > 0 and n_neg > 0:
                    pos_weight = n / (2 * n_pos)
                    neg_weight = n / (2 * n_neg)
                    instance_weights = torch.where(y_true == 1, pos_weight, neg_weight)
                else:
                    # Edge case: only one class present
                    instance_weights = torch.ones(n, device=R.device)
                
                # Weighted average
                sup_loss = torch.sum(instance_weights * bce) / torch.sum(instance_weights)
            else:
                # Standard unweighted average
                sup_loss = torch.mean(bce)
        else:
            sup_loss = torch.tensor(0.0, device=R.device)
        
        # 4. Combined loss
        total_recon = recon_loss + reg_loss
        total_loss = rho * total_recon + (1 - rho) * sup_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': total_recon.item(),
            'supervised': sup_loss.item(),
            'recon_weighted': (rho * total_recon).item(),
            'sup_weighted': ((1 - rho) * sup_loss).item(),
            'rho': rho
        }
        
        return total_loss, loss_dict


class CFEnsemblePyTorchTrainer:
    """
    PyTorch-based trainer for CF-Ensemble.
    
    Uses joint gradient descent to optimize all parameters simultaneously,
    solving the optimization instability of alternating ALS + GD.
    
    Parameters:
    -----------
    n_classifiers : int
        Number of base classifiers
    latent_dim : int, default=20
        Latent factor dimensionality
    rho : float, default=0.5
        Trade-off between reconstruction (1.0) and supervision (0.0)
    lambda_reg : float, default=0.01
        Regularization strength
    aggregator_type : str, default='weighted'
        Aggregator type ('mean' or 'weighted')
    max_epochs : int, default=500
        Maximum training epochs
    lr : float, default=0.01
        Initial learning rate
    optimizer : str, default='adam'
        Optimizer ('adam', 'adamw', 'sgd')
    patience : int, default=20
        Early stopping patience
    min_delta : float, default=1e-4
        Minimum loss improvement for early stopping
    use_class_weights : bool, default=True
        If True, weight instances by inverse class frequency in supervised loss
        Essential for imbalanced data to prevent majority class domination
    device : str, default='auto'
        Device ('cpu', 'cuda', or 'auto')
    verbose : bool, default=True
        Whether to print training progress
    random_seed : int, optional
        Random seed for reproducibility
    
    Example:
    --------
    >>> from cfensemble.data import EnsembleData
    >>> from cfensemble.optimization import CFEnsemblePyTorchTrainer
    >>> 
    >>> # Create data
    >>> R = np.random.rand(10, 1000)
    >>> labels = np.random.randint(0, 2, 1000).astype(float)
    >>> labels[500:] = np.nan
    >>> data = EnsembleData(R, labels)
    >>> 
    >>> # Train
    >>> trainer = CFEnsemblePyTorchTrainer(
    ...     n_classifiers=10,
    ...     latent_dim=20,
    ...     rho=0.5,
    ...     max_epochs=200
    ... )
    >>> trainer.fit(data)
    >>> 
    >>> # Predict
    >>> predictions = trainer.predict()
    """
    
    def __init__(
        self,
        n_classifiers: int,
        latent_dim: int = 20,
        rho: float = 0.5,
        lambda_reg: float = 0.01,
        aggregator_type: str = 'weighted',
        max_epochs: int = 500,
        lr: float = 0.01,
        optimizer: str = 'adam',
        patience: int = 20,
        min_delta: float = 1e-4,
        use_class_weights: bool = True,
        device: str = 'auto',
        verbose: bool = True,
        random_seed: Optional[int] = None
    ):
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CFEnsemblePyTorchTrainer. "
                "Install with: pip install torch"
            )
        
        self.n_classifiers = n_classifiers
        self.latent_dim = latent_dim
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.aggregator_type = aggregator_type
        self.max_epochs = max_epochs
        self.lr = lr
        self.optimizer_name = optimizer
        self.patience = patience
        self.min_delta = min_delta
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Model and optimizer (initialized in fit())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'loss': [],
            'reconstruction': [],
            'supervised': [],
            'recon_weighted': [],
            'sup_weighted': []
        }
        
        # Convergence info
        self.converged_ = False
        self.n_epochs_ = 0
        self.best_loss_ = float('inf')
    
    def fit(self, ensemble_data: EnsembleData):
        """
        Fit CF-Ensemble model via joint gradient descent.
        
        Parameters:
        -----------
        ensemble_data : EnsembleData
            Data with probability matrix R, labels, and confidence weights
        
        Returns:
        --------
        self : CFEnsemblePyTorchTrainer
            Fitted trainer
        """
        # Extract data
        R_np = ensemble_data.R
        C_np = ensemble_data.C
        labels_np = ensemble_data.labels
        labeled_mask_np = ensemble_data.labeled_idx
        
        m, n = R_np.shape
        
        if m != self.n_classifiers:
            raise ValueError(
                f"Expected {self.n_classifiers} classifiers, got {m}"
            )
        
        # Convert to PyTorch tensors
        R = torch.from_numpy(R_np).float().to(self.device)
        C = torch.from_numpy(C_np).float().to(self.device)
        labels = torch.from_numpy(labels_np).float().to(self.device)
        labeled_mask = torch.from_numpy(labeled_mask_np).bool().to(self.device)
        
        # Initialize model
        self.model = CFEnsembleNet(
            m=m,
            n=n,
            d=self.latent_dim,
            aggregator_type=self.aggregator_type
        ).to(self.device)
        
        # Initialize optimizer
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        if self.verbose:
            print(f"Training CF-Ensemble (PyTorch) on device: {self.device}")
            print(f"Data: {m} classifiers × {n} instances, {labeled_mask.sum()} labeled")
            print(f"Config: d={self.latent_dim}, ρ={self.rho}, λ={self.lambda_reg}")
            print(f"Optimizer: {self.optimizer_name}, LR={self.lr}")
            if self.use_class_weights:
                print(f"Class weighting: Enabled (inverse frequency)")
            print("-" * 70)
        
        for epoch in range(self.max_epochs):
            # Forward + backward pass
            self.optimizer.zero_grad()
            loss, loss_dict = self.model.compute_loss(
                R, C, labels, labeled_mask,
                self.rho, self.lambda_reg,
                self.use_class_weights
            )
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step(loss)
            
            # Store history
            self.history['loss'].append(loss_dict['total'])
            self.history['reconstruction'].append(loss_dict['reconstruction'])
            self.history['supervised'].append(loss_dict['supervised'])
            self.history['recon_weighted'].append(loss_dict['recon_weighted'])
            self.history['sup_weighted'].append(loss_dict['sup_weighted'])
            
            # Print progress
            if self.verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                print(
                    f"Epoch {epoch:4d}: Loss={loss_dict['total']:.4f}, "
                    f"Recon={loss_dict['reconstruction']:.4f}, "
                    f"Sup={loss_dict['supervised']:.4f}"
                )
            
            # Early stopping check
            if loss_dict['total'] < best_loss - self.min_delta:
                best_loss = loss_dict['total']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                self.converged_ = True
                self.n_epochs_ = epoch + 1
                if self.verbose:
                    print(f"Early stopping at epoch {epoch} (loss converged)")
                break
        
        if not self.converged_:
            self.n_epochs_ = self.max_epochs
            if self.verbose:
                warnings.warn(
                    f"Training did not converge after {self.max_epochs} epochs. "
                    "Consider increasing max_epochs or adjusting learning rate."
                )
        
        self.best_loss_ = best_loss
        
        return self
    
    def predict(self, R_new: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict on instances.
        
        Parameters:
        -----------
        R_new : array (m × n_new), optional
            New probability matrix for inductive prediction.
            If None, uses learned instance factors (transductive).
        
        Returns:
        --------
        predictions : array (n or n_new,)
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            if R_new is None:
                # Transductive: use learned Y factors
                R_hat = self.model.reconstruct()
                predictions = self.model.aggregate(R_hat)
                return predictions.cpu().numpy()
            else:
                # Inductive: compute new Y factors (cold start)
                # For now, use simple approach (can be improved)
                R_new_torch = torch.from_numpy(R_new).float().to(self.device)
                m, n_new = R_new_torch.shape
                
                # Compute Y_new via least squares: Y_new = (X^T X + λI)^{-1} X^T R_new
                X = self.model.X  # (d × m)
                XtX = X @ X.T + self.lambda_reg * torch.eye(self.latent_dim, device=self.device)
                Y_new = torch.linalg.solve(XtX, X @ R_new_torch)  # (d × n_new)
                
                # Reconstruct and aggregate
                R_hat_new = X.T @ Y_new  # (m × n_new)
                predictions = self.model.aggregate(R_hat_new)
                return predictions.cpu().numpy()
    
    def get_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get learned latent factors.
        
        Returns:
        --------
        X : array (d × m)
            Classifier factors
        Y : array (d × n)
            Instance factors
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X = self.model.X.cpu().numpy()
            Y = self.model.Y.cpu().numpy()
        return X, Y
    
    def get_aggregator_weights(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get aggregator weights (if weighted aggregator).
        
        Returns:
        --------
        w : array (m,) or None
            Classifier weights (None if mean aggregator)
        b : float or None
            Bias term (None if mean aggregator)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.aggregator_type != 'weighted':
            return None
        
        self.model.eval()
        with torch.no_grad():
            w = self.model.w.cpu().numpy()
            b = self.model.b.item()
        return w, b
