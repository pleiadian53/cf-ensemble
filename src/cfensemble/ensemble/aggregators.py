"""
Aggregation Functions for CF-Ensemble

This module implements aggregation functions that map reconstructed probabilities
from multiple classifiers to final predictions.

Based on: docs/methods/cf_ensemble_optimization_objective_tutorial.md Section 7
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


class BaseAggregator(ABC):
    """
    Base class for aggregation functions.
    
    An aggregator takes reconstructed probabilities from m classifiers
    and produces a final prediction.
    """
    
    @abstractmethod
    def predict(self, r_hat: np.ndarray) -> np.ndarray:
        """
        Map reconstructed probabilities to final predictions.
        
        Parameters:
            r_hat: Reconstructed probabilities (m × n_batch)
                - m = number of classifiers
                - n_batch = number of instances to predict
        
        Returns:
            predictions: Final probabilities (n_batch,)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        lr: float
    ):
        """
        Update aggregator parameters via gradient descent.
        
        Parameters:
            X: Classifier factors (d × m)
            Y: Instance factors (d × n)
            labeled_idx: Boolean mask for labeled instances
            labels: Ground truth labels (n,)
            lr: Learning rate
        """
        pass
    
    def get_params(self) -> dict:
        """Get aggregator parameters (for saving/loading)."""
        return {}
    
    def set_params(self, params: dict):
        """Set aggregator parameters (for saving/loading)."""
        pass


class MeanAggregator(BaseAggregator):
    """
    Simple averaging aggregator.
    
    g(r̂) = (1/m) Σ r̂_u
    
    No learnable parameters. Treats all classifiers equally.
    
    Example:
        >>> agg = MeanAggregator()
        >>> r_hat = np.random.rand(10, 100)  # 10 classifiers, 100 instances
        >>> predictions = agg.predict(r_hat)
        >>> predictions.shape
        (100,)
    """
    
    def predict(self, r_hat: np.ndarray) -> np.ndarray:
        """
        Average predictions across classifiers.
        
        Parameters:
            r_hat: Reconstructed probabilities (m × n_batch)
        
        Returns:
            predictions: Mean probabilities (n_batch,)
        """
        return np.mean(r_hat, axis=0)
    
    def update(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        lr: float
    ):
        """No parameters to update."""
        pass


class WeightedAggregator(BaseAggregator):
    """
    Weighted averaging with learnable weights.
    
    g(r̂) = σ(w^T r̂ + b)
    
    Where σ is sigmoid, w are classifier weights, and b is bias.
    
    This allows the system to learn which classifiers to trust more,
    globally across all instances.
    
    Parameters:
        n_classifiers: Number of base classifiers
        init_uniform: If True, initialize weights uniformly (1/m each)
                     If False, initialize randomly
    
    Example:
        >>> agg = WeightedAggregator(n_classifiers=10)
        >>> r_hat = np.random.rand(10, 100)
        >>> predictions = agg.predict(r_hat)
        >>> 
        >>> # Update weights via gradient descent
        >>> X = np.random.randn(20, 10)
        >>> Y = np.random.randn(20, 1000)
        >>> labels = np.random.randint(0, 2, 1000).astype(float)
        >>> labeled_idx = np.arange(500) < 250
        >>> agg.update(X, Y, labeled_idx, labels, lr=0.01)
    """
    
    def __init__(self, n_classifiers: int, init_uniform: bool = True):
        """
        Initialize weighted aggregator.
        
        Parameters:
            n_classifiers: Number of base classifiers
            init_uniform: If True, initialize w = [1/m, ..., 1/m], b = 0
                         If False, initialize randomly
        """
        self.n_classifiers = n_classifiers
        
        if init_uniform:
            self.w = np.ones(n_classifiers) / n_classifiers
            self.b = 0.0
        else:
            self.w = np.random.randn(n_classifiers) * 0.01
            self.b = 0.0
    
    def predict(self, r_hat: np.ndarray) -> np.ndarray:
        """
        Weighted average with sigmoid activation.
        
        Parameters:
            r_hat: Reconstructed probabilities (m × n_batch)
        
        Returns:
            predictions: Weighted probabilities (n_batch,)
        """
        logits = self.w @ r_hat + self.b  # (n_batch,)
        return sigmoid(logits)
    
    def update(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        lr: float,
        use_class_weights: bool = True,
        focal_gamma: float = 0.0
    ):
        """
        Update weights via gradient descent on cross-entropy or focal loss.
        
        Supports two orthogonal techniques for handling imbalanced/difficult data:
        
        1. **Class weighting** (use_class_weights=True):
           - Balances contribution from each class (addresses class imbalance)
           - Uses inverse class frequency: weight = n / (2 * n_class)
           - Prevents majority class from dominating gradients
        
        2. **Focal loss** (focal_gamma > 0):
           - Down-weights easy examples, focuses on hard ones
           - Modulating factor: (1 - p_t)^gamma
           - Addresses easy vs. hard example imbalance
        
        These can be combined: class weighting handles class imbalance,
        focal loss handles example difficulty.
        
        Gradient formula:
            focal_weight = (1 - p_t)^gamma  where p_t = p if y=1, else (1-p)
            total_weight = class_weight * focal_weight
            ∂L/∂w = Σ total_weight[i] * (ŷ[i] - y[i]) * r̂[i] / Σ total_weight
            ∂L/∂b = Σ total_weight[i] * (ŷ[i] - y[i]) / Σ total_weight
        
        Parameters:
            X: Classifier factors (d × m)
            Y: Instance factors (d × n)
            labeled_idx: Boolean mask for labeled instances (n,)
            labels: Ground truth labels (n,)
            lr: Learning rate
            use_class_weights: If True, weight instances by inverse class frequency
                             Recommended for imbalanced data (default: True)
            focal_gamma: Focal loss exponent (0.0 = disabled, 2.0 = standard)
                        Higher gamma down-weights easy examples more strongly
        """
        # Reconstruct probabilities for labeled instances
        R_hat = X.T @ Y[:, labeled_idx]  # (m × n_labeled)
        
        # Get predictions
        y_pred = self.predict(R_hat)  # (n_labeled,)
        y_true = labels[labeled_idx]  # (n_labeled,)
        
        # Compute residuals
        residual = y_pred - y_true  # (n_labeled,)
        
        # Compute instance weights (combination of class and focal weighting)
        n = len(y_true)
        instance_weights = np.ones(n)
        
        # 1. Class weighting (for imbalanced classes)
        if use_class_weights:
            n_pos = np.sum(y_true == 1)
            n_neg = n - n_pos
            
            # Inverse class frequency weights
            # Formula: weight = n / (2 * n_class)
            # This ensures each class contributes equally to the gradient
            if n_pos > 0 and n_neg > 0:
                pos_weight = n / (2 * n_pos)
                neg_weight = n / (2 * n_neg)
                instance_weights = np.where(y_true == 1, pos_weight, neg_weight)
        
        # 2. Focal loss modulation (for hard examples)
        if focal_gamma > 0:
            # Compute p_t: probability of true class
            # p_t = p if y=1, else (1-p)
            p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
            
            # Focal weight: (1 - p_t)^gamma
            # Easy examples (high p_t) get low weight
            # Hard examples (low p_t) get high weight
            focal_weight = np.power(1 - p_t, focal_gamma)
            
            # Combine with class weights
            instance_weights = instance_weights * focal_weight
        
        # Compute weighted gradients
        if use_class_weights or focal_gamma > 0:
            # Normalize by sum of weights
            weighted_residual = residual * instance_weights
            grad_w = (R_hat @ weighted_residual) / np.sum(instance_weights)
            grad_b = np.sum(weighted_residual) / np.sum(instance_weights)
        else:
            # Standard (unweighted) gradient
            grad_w = (R_hat @ residual) / len(residual)
            grad_b = np.mean(residual)
        
        # Gradient descent update
        self.w -= lr * grad_w
        self.b -= lr * grad_b
    
    def get_params(self) -> dict:
        """Get weights and bias."""
        return {'w': self.w.copy(), 'b': self.b}
    
    def set_params(self, params: dict):
        """Set weights and bias."""
        self.w = params['w'].copy()
        self.b = params['b']
    
    def get_weights(self) -> np.ndarray:
        """Get classifier weights (for analysis)."""
        return self.w.copy()
    
    def get_normalized_weights(self) -> np.ndarray:
        """
        Get normalized weights (sum to 1).
        
        Useful for interpreting classifier importance.
        """
        w_abs = np.abs(self.w)
        return w_abs / np.sum(w_abs)


class InstanceDependentAggregator(BaseAggregator):
    """
    Instance-dependent weighted aggregator (mixture of experts).
    
    α_ui = softmax_u(g(features))
    ŷ_i = Σ α_ui · r̂_ui
    
    This is more powerful than WeightedAggregator as it learns
    which classifiers to trust for each specific instance.
    
    Note: More complex, requires more careful tuning to avoid overfitting.
    Recommended only after basic methods are working.
    
    Parameters:
        n_classifiers: Number of base classifiers
        hidden_dim: Hidden layer size for gating network
    """
    
    def __init__(self, n_classifiers: int, hidden_dim: int = 32):
        """Initialize instance-dependent aggregator."""
        self.n_classifiers = n_classifiers
        self.hidden_dim = hidden_dim
        
        # Gating network parameters
        # Input: reconstructed probabilities (m,)
        # Output: weights over classifiers (m,)
        self.W1 = np.random.randn(hidden_dim, n_classifiers) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(n_classifiers, hidden_dim) * 0.01
        self.b2 = np.zeros(n_classifiers)
    
    def _gating_network(self, r_hat: np.ndarray) -> np.ndarray:
        """
        Compute instance-specific weights.
        
        Parameters:
            r_hat: Reconstructed probabilities (m × n_batch)
        
        Returns:
            alpha: Instance-specific weights (m × n_batch)
        """
        # Hidden layer
        h = np.tanh(self.W1 @ r_hat + self.b1[:, None])  # (hidden_dim × n_batch)
        
        # Output layer
        logits = self.W2 @ h + self.b2[:, None]  # (m × n_batch)
        
        # Softmax over classifiers
        logits_max = np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        alpha = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        
        return alpha
    
    def predict(self, r_hat: np.ndarray) -> np.ndarray:
        """
        Weighted average with instance-dependent weights.
        
        Parameters:
            r_hat: Reconstructed probabilities (m × n_batch)
        
        Returns:
            predictions: Weighted probabilities (n_batch,)
        """
        alpha = self._gating_network(r_hat)  # (m × n_batch)
        return np.sum(alpha * r_hat, axis=0)  # (n_batch,)
    
    def update(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        labeled_idx: np.ndarray,
        labels: np.ndarray,
        lr: float
    ):
        """
        Update gating network via backpropagation.
        
        Note: This is more complex. For now, implemented as placeholder.
        Full implementation would require careful gradient computation.
        """
        # TODO: Implement full backpropagation
        # For now, just pass (use mean or weighted aggregator first)
        pass
    
    def get_params(self) -> dict:
        """Get gating network parameters."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_params(self, params: dict):
        """Set gating network parameters."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
