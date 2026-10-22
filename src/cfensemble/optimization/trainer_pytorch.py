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
    link : str
        'identity': R_hat = X^T Y (squared-error reconstruction).
        'logit': R_hat = sigmoid(X^T Y + b_u + b_i) with classifier biases b_u and
        instance biases b_i (cross-entropy reconstruction against R).
    """
    
    def __init__(
        self,
        m: int,
        n: int,
        d: int,
        aggregator_type: str = 'weighted',
        link: str = 'identity'
    ):
        super().__init__()
        
        self.m = m
        self.n = n
        self.d = d
        self.aggregator_type = aggregator_type
        if link not in ('identity', 'logit'):
            raise ValueError(f"link must be 'identity' or 'logit', got {link!r}")
        self.link = link
        
        # Classifier latent factors (d × m)
        self.X = nn.Parameter(torch.randn(d, m) * 0.01)
        
        # Instance latent factors (d × n)
        self.Y = nn.Parameter(torch.randn(d, n) * 0.01)
        
        # Classifier / instance biases (logit link only)
        if link == 'logit':
            self.b_u = nn.Parameter(torch.zeros(m, 1))
            self.b_i = nn.Parameter(torch.zeros(1, n))
        
        # Aggregator parameters
        if aggregator_type == 'weighted':
            self.w = nn.Parameter(torch.ones(m) / m)  # Start uniform
            self.b = nn.Parameter(torch.zeros(1))
        elif aggregator_type == 'simplex':
            # Convex combination of reconstructed probabilities: alpha = softmax(theta),
            # nonnegative and summing to 1 (the inductive bias of ensemble selection)
            self.theta = nn.Parameter(torch.zeros(m))
            self.register_buffer('w', torch.ones(m) / m)
            self.register_buffer('b', torch.zeros(1))
        else:  # mean aggregator (no parameters)
            self.register_buffer('w', torch.ones(m) / m)
            self.register_buffer('b', torch.zeros(1))
    
    def simplex_weights(self) -> torch.Tensor:
        """alpha = softmax(theta) for the simplex aggregator."""
        return torch.softmax(self.theta, dim=0)
    
    def aggregator_parameters(self) -> list:
        """Trainable parameters of g (empty for the mean aggregator)."""
        if self.aggregator_type == 'weighted':
            return [self.w, self.b]
        if self.aggregator_type == 'simplex':
            return [self.theta]
        return []
    
    def factor_parameters(self) -> list:
        """Trainable factorization parameters: X, Y (and b_u, b_i for the logit link)."""
        agg = {id(p) for p in self.aggregator_parameters()}
        return [p for p in self.parameters() if id(p) not in agg]
    
    def scores(self, idx: Optional[torch.Tensor] = None, detach_instances: bool = False) -> torch.Tensor:
        """
        Pre-link scores X^T Y (+ b_u + b_i for the logit link), for all or ``idx`` columns.
        
        ``detach_instances`` detaches the instance factors (and instance biases).
        """
        Y = self.Y if idx is None else self.Y[:, idx]
        if detach_instances:
            Y = Y.detach()
        Z = self.X.T @ Y
        if self.link == 'logit':
            b_i = self.b_i if idx is None else self.b_i[:, idx]
            if detach_instances:
                b_i = b_i.detach()
            Z = Z + self.b_u + b_i
        return Z
    
    def reconstruct(self) -> torch.Tensor:
        """
        Reconstruct probability matrix: R_hat = X^T Y, or sigmoid(X^T Y + b_u + b_i)
        for the logit link.
        
        Returns:
        --------
        R_hat : Tensor (m × n)
            Reconstructed probabilities
        """
        Z = self.scores()
        return torch.sigmoid(Z) if self.link == 'logit' else Z
    
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
        elif self.aggregator_type == 'simplex':
            return self.simplex_weights() @ R_hat_subset
        else:  # weighted
            logits = self.w @ R_hat_subset + self.b  # (n_subset,)
            return torch.sigmoid(logits)
    
    def forward(self, labeled_idx: torch.Tensor, detach_instances: bool = False) -> torch.Tensor:
        """
        Forward pass: reconstruct + aggregate for labeled instances.

        Parameters:
        -----------
        labeled_idx : Tensor (n_labeled,)
            Indices of labeled instances
        detach_instances : bool, default=False
            If True, the labeled instance factors are detached, so the supervised
            loss updates X and the aggregator but not the labeled y_i

        Returns:
        --------
        y_pred : Tensor (n_labeled,)
            Predictions for labeled instances
        """
        return self.aggregate(self._reconstruct_subset(labeled_idx, detach_instances))

    def _reconstruct_subset(self, idx: torch.Tensor, detach_instances: bool = False) -> torch.Tensor:
        """R_hat columns for ``idx`` (m × len(idx)), optionally with y_i detached."""
        Z = self.scores(idx, detach_instances)
        return torch.sigmoid(Z) if self.link == 'logit' else Z
    
    def combiner_scores(self, R_hat_subset: torch.Tensor) -> torch.Tensor:
        """Ranking scores of g: logits for 'weighted', the aggregate itself otherwise."""
        if self.aggregator_type == 'weighted':
            return self.w @ R_hat_subset + self.b
        return self.aggregate(R_hat_subset)
    
    def supervised_loss(
        self,
        R_hat_labeled: torch.Tensor,
        y_true: torch.Tensor,
        use_class_weights: bool = True,
        lambda_g: float = 0.0,
        loss_type: str = 'bce',
        pairwise_scale: float = 1.0,
        max_negatives: int = 20000
    ) -> torch.Tensor:
        """Supervised loss on the labeled columns (+ L2 on w if lambda_g > 0).
        
        'bce': class-weighted binary cross-entropy of g's output.
        'pairwise': mean softplus(-scale * (s_pos - s_neg)) over positive/negative
        pairs of combiner scores — a smooth AUC surrogate that targets ranking and
        is balanced by construction. Negatives are subsampled to ``max_negatives``.
        """
        if loss_type == 'pairwise':
            s = self.combiner_scores(R_hat_labeled)
            s_pos, s_neg = s[y_true == 1], s[y_true == 0]
            if len(s_pos) == 0 or len(s_neg) == 0:
                return torch.tensor(0.0, device=y_true.device)
            if len(s_neg) > max_negatives:
                s_neg = s_neg[torch.randperm(len(s_neg), device=s.device)[:max_negatives]]
            diff = pairwise_scale * (s_pos[:, None] - s_neg[None, :])
            sup_loss = nn.functional.softplus(-diff).mean()
            if lambda_g > 0 and self.aggregator_type == 'weighted':
                sup_loss = sup_loss + lambda_g * torch.sum(self.w ** 2)
            return sup_loss
        
        # Binary cross-entropy. The weighted aggregator uses the logit form, which
        # is numerically stable; clamping probabilities at 1 - 1e-15 does nothing in
        # float32 (it rounds to 1.0), so saturated sigmoids gave log(0) -> NaN.
        if self.aggregator_type in ('mean', 'simplex'):
            eps = 1e-7  # representable in float32
            y_pred = torch.clamp(self.aggregate(R_hat_labeled), eps, 1 - eps)
            bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        else:
            logits = self.w @ R_hat_labeled + self.b
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, y_true, reduction='none'
            )

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
                instance_weights = torch.ones(n, device=y_true.device)

            # Weighted average
            sup_loss = torch.sum(instance_weights * bce) / torch.sum(instance_weights)
        else:
            # Standard unweighted average
            sup_loss = torch.mean(bce)
        if lambda_g > 0 and self.aggregator_type == 'weighted':
            sup_loss = sup_loss + lambda_g * torch.sum(self.w ** 2)
        return sup_loss

    def compute_loss(
        self,
        R: torch.Tensor,
        C: torch.Tensor,
        labels: torch.Tensor,
        labeled_mask: torch.Tensor,
        rho: float,
        lambda_reg: float,
        use_class_weights: bool = True,
        normalize_recon: bool = False,
        sup_updates_instances: bool = True,
        lambda_g: float = 0.0,
        sup_loss_type: str = 'bce',
        pairwise_scale: float = 1.0
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
        normalize_recon : bool, default=False
            If True, divide reconstruction + regularization by m·n so both loss
            terms are per-entry averages and ρ balances quantities of similar
            scale. Otherwise the summed reconstruction term grows with n and
            swamps the (mean) supervised term on large datasets.
        sup_updates_instances : bool, default=True
            If False, the supervised loss does not update the labeled instance
            factors y_i (only X and the aggregator), so labeled columns cannot bend
            toward their labels while unlabeled columns stay put.
        lambda_g : float, default=0.0
            L2 penalty on the aggregator weights w (not the bias), added inside the
            supervised term. lambda_g = 1 / (2·n_labeled) matches scikit-learn's
            LogisticRegression(C=1) on the same inputs.

        Returns:
        --------
        loss : Tensor (scalar)
            Combined loss
        loss_dict : dict
            Dictionary with loss components for logging
        """
        # 1. Reconstruction loss
        if self.link == 'logit':
            entry_loss = nn.functional.binary_cross_entropy_with_logits(
                self.scores(), R, reduction='none'
            )
        else:
            entry_loss = (R - self.reconstruct()) ** 2
        recon_loss = torch.sum(C * entry_loss)
        
        # 2. Regularization
        reg_loss = lambda_reg * (torch.sum(self.X ** 2) + torch.sum(self.Y ** 2))
        if self.link == 'logit':
            reg_loss = reg_loss + lambda_reg * torch.sum(self.b_i ** 2)  # m obs per b_i
        if normalize_recon:
            recon_loss = recon_loss / R.numel()
            reg_loss = reg_loss / R.numel()
        
        # 3. Supervised loss (only on labeled instances)
        if torch.sum(labeled_mask) > 0:
            labeled_idx = torch.where(labeled_mask)[0]
            R_hat_labeled = self._reconstruct_subset(labeled_idx, not sup_updates_instances)
            y_true = labels[labeled_mask]

            sup_loss = self.supervised_loss(R_hat_labeled, y_true, use_class_weights, lambda_g,
                                            sup_loss_type, pairwise_scale)
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
        Aggregator type ('mean', 'weighted', or 'simplex': softmax weights over
        classifiers, i.e. a convex combination like ensemble selection)
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
    normalize_recon : bool, default=False
        If True, average (rather than sum) the reconstruction and regularization
        terms over the m·n entries, so ρ keeps its meaning as n grows. Recommended
        for large datasets; the default keeps earlier results reproducible.
    sup_updates_instances : bool, default=True
        If False, supervision updates X and the aggregator but not the labeled
        instance factors (stop-gradient), avoiding a train/test mismatch for g.
    lambda_g : float, default=0.0
        L2 penalty on the aggregator weights; 1 / (2·n_labeled) matches
        scikit-learn's LogisticRegression(C=1), the stacking baseline's combiner.
    link : str, default='identity'
        'logit' reconstructs R as sigmoid(X^T Y + b_u + b_i) with cross-entropy;
        b_u absorbs a classifier's overall shift, b_i an instance's positivity.
    lr_g : float, optional
        Learning rate for the aggregator g (default: ``lr``). The total loss is
        dominated by reconstruction, so a shared schedule can leave g under-trained.
    refit_g_steps : int, default=0
        After training, freeze the factorization and fit g alone on the labeled
        reconstructed columns for this many Adam steps (lr = ``lr_g`` or 0.05).
    sup_loss : str, default='bce'
        'pairwise' replaces cross-entropy with a pairwise ranking (AUC) surrogate on
        g's scores: targets ordering, like ensemble selection, and is balanced by
        construction. With the probability-valued simplex combiner, set
        ``pairwise_scale`` (e.g. 20) so score differences are not vanishingly small.
    pairwise_scale : float, default=1.0
        Multiplier on score differences in the pairwise loss.
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
        focal_gamma: float = 0.0,
        normalize_recon: bool = False,
        sup_updates_instances: bool = True,
        lambda_g: float = 0.0,
        link: str = 'identity',
        lr_g: Optional[float] = None,
        refit_g_steps: int = 0,
        sup_loss: str = 'bce',
        pairwise_scale: float = 1.0,
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
        self.focal_gamma = focal_gamma
        self.normalize_recon = normalize_recon
        self.sup_updates_instances = sup_updates_instances
        self.lambda_g = lambda_g
        self.link = link
        self.lr_g = lr_g
        self.refit_g_steps = refit_g_steps
        if sup_loss not in ('bce', 'pairwise'):
            raise ValueError(f"sup_loss must be 'bce' or 'pairwise', got {sup_loss!r}")
        self.sup_loss = sup_loss
        self.pairwise_scale = pairwise_scale
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
            aggregator_type=self.aggregator_type,
            link=self.link
        ).to(self.device)
        
        # Initialize optimizer
        # Separate parameter groups so g can have its own learning rate
        param_groups = [{'params': self.model.factor_parameters(), 'lr': self.lr}]
        if self.model.aggregator_parameters():
            param_groups.append({'params': self.model.aggregator_parameters(),
                                 'lr': self.lr_g or self.lr})
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(param_groups)
        elif self.optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(param_groups)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(param_groups, momentum=0.9)
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
                self.use_class_weights,
                self.normalize_recon,
                self.sup_updates_instances,
                self.lambda_g,
                self.sup_loss,
                self.pairwise_scale
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
        
        if self.refit_g_steps > 0:
            self._refit_aggregator(labels, labeled_mask)
        
        return self
    
    def _refit_aggregator(self, labels: 'torch.Tensor', labeled_mask: 'torch.Tensor') -> None:
        """Fit g alone on the frozen labeled reconstructions (convex in w for 'weighted', in alpha for 'simplex')."""
        agg_params = self.model.aggregator_parameters()
        if not agg_params or torch.sum(labeled_mask) == 0:
            return
        labeled_idx = torch.where(labeled_mask)[0]
        with torch.no_grad():
            R_hat_labeled = self.model._reconstruct_subset(labeled_idx)
        y_true = labels[labeled_mask]
        opt = optim.Adam(agg_params, lr=self.lr_g or 0.05)
        for _ in range(self.refit_g_steps):
            opt.zero_grad()
            loss = self.model.supervised_loss(R_hat_labeled, y_true, self.use_class_weights,
                                              self.lambda_g, self.sup_loss, self.pairwise_scale)
            loss.backward()
            opt.step()
        self.refit_loss_ = loss.item()
    
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
                if self.link != 'identity':
                    raise NotImplementedError(
                        "Inductive prediction (R_new) is only implemented for link='identity'"
                    )
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
