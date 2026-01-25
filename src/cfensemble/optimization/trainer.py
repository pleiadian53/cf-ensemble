"""
CF-Ensemble Trainer with ALS Optimization (APPROXIMATION)

This module implements CFEnsembleTrainer using Alternating Least Squares (ALS).

IMPORTANT - This is an APPROXIMATION of the true combined loss:
    L_CF = ρ·L_recon(X,Y) + (1-ρ)·L_sup(X,Y,θ)

Why approximation?
- ALS only optimizes QUADRATIC objectives (closed-form solutions)
- Supervised loss L_sup contains sigmoid + log (non-quadratic)
- Cannot derive closed-form ALS for full combined loss

Approximation strategy:
1. Use label-aware confidence: c_ui upweights predictions matching labels
2. ALS minimizes: Σ c_ui(r_ui - x_u^T y_i)² (indirectly encourages supervision)
3. Separate aggregator gradient descent (coordinates with ALS)

This is analogous to how VAE uses reparameterization trick + gradient descent,
not closed-form, even though both have reconstruction + regularization terms.

For EXACT optimization: Use CFEnsemblePyTorchTrainer (recommended).

Trade-offs:
+ Fast per iteration (O(d³) closed-form)
+ Works without autodiff framework
- Approximate (not exact combined gradient)
- Potential alternating optimization instability
"""

import numpy as np
from typing import Optional, Dict, Any, Literal
import warnings

from ..data.ensemble_data import EnsembleData
from ..ensemble.aggregators import BaseAggregator, MeanAggregator, WeightedAggregator
from ..objectives.losses import combined_loss
from .als import update_classifier_factors, update_instance_factors


class CFEnsembleTrainer:
    """
    Main training loop for CF-Ensemble.
    
    Implements the alternating optimization algorithm:
    1. Update X (classifier factors) via ALS
    2. Update Y (instance factors) via ALS
    3. Update aggregator parameters via gradient descent
    4. Monitor combined loss for convergence
    
    The trainer balances reconstruction fidelity with supervised prediction
    through the parameter ρ ∈ [0,1]:
    - ρ=1.0: Pure matrix reconstruction (no supervision)
    - ρ=0.5: Balanced (recommended starting point)
    - ρ=0.0: Pure supervised learning (no reconstruction constraint)
    
    Attributes:
        n_classifiers: Number of base classifiers (m)
        latent_dim: Dimensionality of latent space (d)
        rho: Balance parameter for combined loss
        lambda_reg: L2 regularization strength
        max_iter: Maximum training iterations
        tol: Convergence tolerance
        aggregator: Aggregation function (mean, weighted, etc.)
        X: Classifier latent factors (d × m)
        Y: Instance latent factors (d × n)
        history: Training history (loss curves)
    
    Example:
        >>> from cfensemble.data import EnsembleData
        >>> from cfensemble.optimization import CFEnsembleTrainer
        >>> 
        >>> # Create data
        >>> data = EnsembleData(R, labels)
        >>> 
        >>> # Train model
        >>> trainer = CFEnsembleTrainer(
        ...     n_classifiers=R.shape[0],
        ...     latent_dim=20,
        ...     rho=0.5,  # Balance reconstruction + supervision
        ...     lambda_reg=0.01
        ... )
        >>> trainer.fit(data)
        >>> 
        >>> # Predict
        >>> predictions = trainer.predict(data)
        >>> 
        >>> # Analyze
        >>> print(f"Final loss: {trainer.history['loss'][-1]:.4f}")
        >>> trainer.plot_loss_curves()
    """
    
    def __init__(
        self,
        n_classifiers: int,
        latent_dim: int = 20,
        rho: float = 0.5,
        lambda_reg: float = 0.01,
        aggregator_type: Literal['mean', 'weighted'] = 'weighted',
        aggregator_lr: float = 0.01,
        max_iter: int = 50,
        tol: float = 1e-4,
        use_label_aware_confidence: bool = True,
        label_aware_alpha: float = 1.0,
        freeze_aggregator_iters: int = 0,
        use_class_weights: bool = True,
        verbose: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize CF-Ensemble trainer.
        
        Parameters:
            n_classifiers: Number of base classifiers (m)
            latent_dim: Dimensionality of latent factors (d)
                       Recommended: 10-50 depending on problem complexity
            rho: Balance between reconstruction and supervision [0, 1]
                 0.0 = pure supervised, 1.0 = pure reconstruction
                 Recommended: 0.5 (balanced) or 0.3-0.7 range
            lambda_reg: L2 regularization strength (λ)
                        Recommended: 0.001-0.1
            aggregator_type: Type of aggregation function
                           'mean': Simple averaging (no parameters)
                           'weighted': Learnable global weights (recommended)
            aggregator_lr: Learning rate for aggregator updates
            max_iter: Maximum number of ALS iterations
            tol: Convergence tolerance (stop if |Δloss| < tol)
            use_label_aware_confidence: Whether to use label-aware confidence
                                       True: Upweight predictions matching labels (approximate supervision)
                                       False: Use certainty-based confidence only
                                       Recommended: True for better approximation
            label_aware_alpha: Strength of label-aware weighting (α ≥ 0)
                              0.0 = no supervision signal
                              1.0 = moderate supervision (recommended)
                              >1.0 = strong supervision emphasis
            freeze_aggregator_iters: Number of initial iterations to freeze aggregator
                                    0 = no freezing (default)
                                    20-100 = recommended range to let X,Y stabilize first
                                    Helps prevent aggregator weight collapse with imbalanced data
            use_class_weights: Whether to use class-weighted gradients in aggregator
                             True: Weight instances by inverse class frequency (recommended)
                             False: Standard unweighted gradients
                             Essential for imbalanced data to prevent majority class domination
            verbose: Whether to print training progress
            random_seed: Random seed for reproducibility
        
        Raises:
            ValueError: If parameters are invalid (rho not in [0,1], etc.)
        """
        # Validate parameters
        if not 0 <= rho <= 1:
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, got {lambda_reg}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if n_classifiers < 1:
            raise ValueError(f"n_classifiers must be positive, got {n_classifiers}")
        
        self.n_classifiers = n_classifiers
        self.latent_dim = latent_dim
        self.rho = rho
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.aggregator_lr = aggregator_lr
        self.use_label_aware_confidence = use_label_aware_confidence
        self.label_aware_alpha = label_aware_alpha
        self.freeze_aggregator_iters = freeze_aggregator_iters
        self.use_class_weights = use_class_weights
        
        # Store random seed for reproducibility
        self.random_seed = random_seed
        
        # Initialize aggregator
        if aggregator_type == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator_type == 'weighted':
            self.aggregator = WeightedAggregator(n_classifiers)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")
        
        # Latent factors (initialized in fit)
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        
        # Training history
        self.history: Dict[str, list] = {
            'loss': [],
            'reconstruction': [],
            'supervised': [],
            'recon_weighted': [],
            'sup_weighted': []
        }
        
        # Convergence flag
        self.converged_ = False
        self.n_iter_ = 0
    
    def fit(self, ensemble_data: EnsembleData) -> 'CFEnsembleTrainer':
        """
        Fit CF-Ensemble model using alternating optimization.
        
        Training procedure:
        1. Initialize X, Y randomly
        2. For each iteration:
           a. Update X given Y (ALS)
           b. Update Y given X (ALS)
           c. Update aggregator parameters (gradient descent)
           d. Compute combined loss
           e. Check convergence
        3. Store training history
        
        Parameters:
            ensemble_data: EnsembleData instance with:
                          - R: Probability matrix (m × n)
                          - labels: Ground truth with NaN for unlabeled
                          - C: Confidence weights (m × n)
        
        Returns:
            self: Fitted trainer (for method chaining)
        
        Raises:
            ValueError: If data shapes are inconsistent
            RuntimeWarning: If training fails to converge
        
        Example:
            >>> trainer = CFEnsembleTrainer(n_classifiers=10, rho=0.5)
            >>> trainer.fit(data)
            Iter 0: Loss=0.4523, Recon=0.3012, Sup=0.6034
            Iter 10: Loss=0.2134, Recon=0.1521, Sup=0.2747
            Converged at iteration 23
        """
        # Extract data
        R = ensemble_data.R
        C = ensemble_data.C
        labels = ensemble_data.labels
        labeled_idx = ensemble_data.labeled_idx
        
        m, n = R.shape
        
        # Validate dimensions
        if m != self.n_classifiers:
            raise ValueError(
                f"Expected {self.n_classifiers} classifiers, "
                f"but R has shape ({m}, {n})"
            )
        
        if np.sum(labeled_idx) == 0:
            warnings.warn(
                "No labeled data provided. Using pure reconstruction (ρ=1.0).",
                RuntimeWarning
            )
            self.rho = 1.0
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Initialize latent factors
        self.X = np.random.randn(self.latent_dim, m) * 0.01
        self.Y = np.random.randn(self.latent_dim, n) * 0.01
        
        # Reset history
        self.history = {
            'loss': [],
            'reconstruction': [],
            'supervised': [],
            'recon_weighted': [],
            'sup_weighted': []
        }
        
        # Compute label-aware confidence if enabled
        if self.use_label_aware_confidence and np.sum(labeled_idx) > 0:
            C = ensemble_data.compute_label_aware_confidence(
                alpha=self.label_aware_alpha
            )
            if self.verbose:
                print(f"Using label-aware confidence (α={self.label_aware_alpha:.2f})")
        # else: C is already set from ensemble_data.C
        
        prev_loss = np.inf
        
        if self.verbose:
            print(f"Starting CF-Ensemble training...")
            print(f"  Data: {m} classifiers × {n} instances")
            print(f"  Labeled: {np.sum(labeled_idx)} ({100*np.mean(labeled_idx):.1f}%)")
            print(f"  Latent dim: {self.latent_dim}")
            print(f"  ρ={self.rho:.2f}, λ={self.lambda_reg:.4f}")
            print(f"  Confidence: {'Label-aware' if self.use_label_aware_confidence else 'Certainty-based'}")
            if self.freeze_aggregator_iters > 0:
                print(f"  Aggregator: Frozen for first {self.freeze_aggregator_iters} iterations")
            if self.use_class_weights:
                print(f"  Class weighting: Enabled (inverse frequency)")
            print()
        
        # Main training loop
        for iteration in range(self.max_iter):
            # 1. Update classifier factors (fix Y)
            self.X = update_classifier_factors(
                self.Y, R, C, self.lambda_reg
            )
            
            # 2. Update instance factors (fix X)
            self.Y = update_instance_factors(
                self.X, R, C, self.lambda_reg
            )
            
            # 3. Update aggregator parameters (fix X, Y)
            # Skip updates during freeze period to let X, Y stabilize
            if iteration >= self.freeze_aggregator_iters and np.sum(labeled_idx) > 0:
                self.aggregator.update(
                    self.X, self.Y, labeled_idx, labels,
                    lr=self.aggregator_lr,
                    use_class_weights=self.use_class_weights
                )
            
            # 4. Compute combined loss
            loss, loss_dict = combined_loss(
                R, self.X, self.Y, C, labels, labeled_idx,
                self.aggregator, self.rho, self.lambda_reg
            )
            
            # Store history
            self.history['loss'].append(loss)
            self.history['reconstruction'].append(loss_dict['reconstruction'])
            self.history['supervised'].append(loss_dict['supervised'])
            self.history['recon_weighted'].append(loss_dict['recon_weighted'])
            self.history['sup_weighted'].append(loss_dict['sup_weighted'])
            
            # 5. Check convergence
            loss_change = abs(prev_loss - loss)
            if loss_change < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                if self.verbose:
                    print(f"✓ Converged at iteration {iteration + 1}")
                    print(f"  Final loss: {loss:.4f}")
                    print(f"  Loss change: {loss_change:.6f} < {self.tol:.6f}")
                break
            
            prev_loss = loss
            
            # Print progress
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(
                    f"Iter {iteration:3d}: "
                    f"Loss={loss:.4f}, "
                    f"Recon={loss_dict['reconstruction']:.4f}, "
                    f"Sup={loss_dict['supervised']:.4f}"
                )
        
        # Check if we hit max iterations
        if not self.converged_:
            self.n_iter_ = self.max_iter
            warnings.warn(
                f"Training did not converge after {self.max_iter} iterations. "
                f"Consider increasing max_iter or adjusting learning parameters.",
                RuntimeWarning
            )
            if self.verbose:
                print(f"\n⚠ Warning: Max iterations ({self.max_iter}) reached without convergence")
        
        if self.verbose:
            print()
        
        return self
    
    def predict(
        self,
        ensemble_data: Optional[EnsembleData] = None,
        R_new: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Two modes:
        1. Transductive: Use fitted Y factors (for training data)
        2. Inductive: Compute Y from new R_new (for test data)
        
        Parameters:
            ensemble_data: EnsembleData instance (uses fitted Y if provided)
            R_new: New probability matrix for inductive prediction (m × n_new)
                   If provided, computes Y_new from R_new
        
        Returns:
            predictions: Predicted probabilities (n,) or (n_new,)
        
        Raises:
            ValueError: If model not fitted or invalid inputs
        
        Example:
            >>> # Transductive prediction (training data)
            >>> y_pred_train = trainer.predict(train_data)
            >>> 
            >>> # Inductive prediction (test data)
            >>> y_pred_test = trainer.predict(R_new=R_test)
        
        Notes:
            - Transductive mode: Uses learned Y directly (best for data seen during training)
            - Inductive mode: Computes Y from R using fitted X (for new data)
            - For true inductive learning, consider training an auxiliary predictor
        """
        if self.X is None or self.Y is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Determine which Y to use
        if R_new is not None:
            # Inductive prediction: compute Y from new R
            if R_new.shape[0] != self.n_classifiers:
                raise ValueError(
                    f"R_new has {R_new.shape[0]} classifiers, "
                    f"expected {self.n_classifiers}"
                )
            
            # Compute Y for new instances using current X
            # This is a simplified inductive approach
            # For each new instance i: y_i = (X^T X + λI)^{-1} X^T r_i
            n_new = R_new.shape[1]
            Y_new = np.zeros((self.latent_dim, n_new))
            
            # Use uniform confidence for new data
            C_new = np.ones_like(R_new)
            
            Y_new = update_instance_factors(
                self.X, R_new, C_new, self.lambda_reg
            )
        else:
            # Transductive prediction: use fitted Y
            Y_new = self.Y
        
        # Reconstruct probabilities
        R_hat = self.X.T @ Y_new  # (m × n) or (m × n_new)
        
        # Aggregate predictions
        predictions = self.aggregator.predict(R_hat)
        
        return predictions
    
    def get_reconstruction(self) -> np.ndarray:
        """
        Get reconstructed probability matrix R_hat = X^T Y.
        
        Returns:
            R_hat: Reconstructed probabilities (m × n)
        
        Example:
            >>> R_hat = trainer.get_reconstruction()
            >>> residuals = data.R - R_hat
            >>> rmse = np.sqrt(np.mean(residuals**2))
        """
        if self.X is None or self.Y is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.X.T @ self.Y
    
    def plot_loss_curves(self, figsize=(12, 4)):
        """
        Plot training loss curves.
        
        Creates three subplots:
        1. Total loss over iterations
        2. Reconstruction and supervised losses
        3. Weighted contributions (ρ·L_recon, (1-ρ)·L_sup)
        
        Parameters:
            figsize: Figure size (width, height)
        
        Example:
            >>> trainer.fit(data)
            >>> trainer.plot_loss_curves()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")
            return
        
        if len(self.history['loss']) == 0:
            print("No training history. Fit the model first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        iterations = range(len(self.history['loss']))
        
        # Total loss
        axes[0].plot(iterations, self.history['loss'], 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Components
        axes[1].plot(iterations, self.history['reconstruction'], 
                    'r-', label='Reconstruction', linewidth=2)
        axes[1].plot(iterations, self.history['supervised'], 
                    'g-', label='Supervised', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Weighted contributions
        axes[2].plot(iterations, self.history['recon_weighted'],
                    'r--', label=f'ρ·L_recon (ρ={self.rho:.2f})', linewidth=2)
        axes[2].plot(iterations, self.history['sup_weighted'],
                    'g--', label=f'(1-ρ)·L_sup', linewidth=2)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Weighted Loss')
        axes[2].set_title('Weighted Contributions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_aggregator_weights(self) -> Optional[np.ndarray]:
        """
        Get learned aggregator weights (if using WeightedAggregator).
        
        Returns:
            weights: Normalized classifier weights (m,) or None
        
        Example:
            >>> weights = trainer.get_aggregator_weights()
            >>> if weights is not None:
            ...     print(f"Most important classifier: {np.argmax(weights)}")
        """
        if isinstance(self.aggregator, WeightedAggregator):
            return self.aggregator.get_normalized_weights()
        return None
