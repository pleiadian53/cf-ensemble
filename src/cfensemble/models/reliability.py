"""
Learned reliability weight models for CF-Ensemble.

This module implements cell-level reliability learning as described in the
Polarity Models Tutorial (docs/methods/polarity_models_tutorial.md).

Key Innovation:
    Instead of treating confidence as a fixed property, learn reliability
    weights from labeled data. This provides m × |L| training examples
    (one per classifier-instance pair) to learn which predictions to trust.

Expected Performance Gain:
    +5-12% ROC-AUC over fixed confidence strategies

Usage:
    >>> from cfensemble.models.reliability import ReliabilityWeightModel
    >>> model = ReliabilityWeightModel(model_type='gbm')
    >>> model.fit(R, labels, labeled_idx)
    >>> W = model.predict(R)  # Use as confidence matrix
"""

from typing import Dict, Optional, Literal
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


ModelType = Literal['gbm', 'rf']


class ReliabilityWeightModel:
    """
    Learn cell-level reliability weights from labeled data.
    
    This model learns to predict how much to trust each prediction r_ui
    based on features extracted from the probability matrix and classifier
    performance patterns.
    
    Training Objective:
        For each cell (u, i) with label y_i:
            target = 1 - |r_ui - y_i|
        
        This is a continuous measure of "correctness" where:
        - target = 1.0 means perfect prediction
        - target = 0.0 means maximally wrong prediction
    
    Features:
        - Raw probability r_ui
        - Distance from threshold |r_ui - 0.5|
        - Instance-level statistics (mean, std, range across classifiers)
        - Optional: Per-classifier performance metrics
    
    Key Benefits:
        1. Cell-level supervision (m × |L| examples, not just |L|)
        2. No pseudo-labels needed on test set
        3. Directly usable as confidence matrix C
        4. Generalizes to unseen instances
    
    Parameters
    ----------
    model_type : {'gbm', 'rf'}, default='gbm'
        Base learning algorithm
        - 'gbm': Gradient Boosting (recommended, captures interactions)
        - 'rf': Random Forest (more robust, less prone to overfitting)
    n_estimators : int, default=100
        Number of trees/boosting iterations
    learning_rate : float, default=0.1
        Learning rate for GBM (ignored for RF)
    max_depth : int, default=3
        Maximum tree depth (controls model complexity)
    random_state : int, default=42
        Random seed for reproducibility
    
    Attributes
    ----------
    model : sklearn estimator
        Fitted regression model
    feature_names_ : list of str
        Names of extracted features
    
    References
    ----------
    See docs/methods/polarity_models_tutorial.md for full motivation and
    comparison with alternative approaches.
    
    Examples
    --------
    Basic usage:
    
    >>> # Train reliability model on labeled data
    >>> rel_model = ReliabilityWeightModel(model_type='gbm')
    >>> rel_model.fit(R, labels, labeled_idx)
    >>> 
    >>> # Predict weights for all cells (including test)
    >>> W = rel_model.predict(R)
    >>> 
    >>> # Use as confidence in CF-Ensemble
    >>> ensemble_data = EnsembleData(R, labels, C=W)
    >>> trainer = CFEnsembleTrainer(rho=0.5)
    >>> trainer.fit(ensemble_data)
    
    With classifier statistics:
    
    >>> # Compute per-classifier metrics
    >>> classifier_stats = {
    >>>     'accuracy': np.array([0.85, 0.78, 0.92, ...]),  # (m,)
    >>>     'auc': np.array([0.88, 0.81, 0.94, ...])
    >>> }
    >>> rel_model.fit(R, labels, labeled_idx, classifier_stats)
    >>> W = rel_model.predict(R, classifier_stats)
    """
    
    def __init__(
        self,
        model_type: ModelType = 'gbm',
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42
    ):
        """Initialize reliability weight model."""
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Create base model
        if model_type == 'gbm':
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                subsample=0.8,  # Prevent overfitting
                min_samples_split=10,
                min_samples_leaf=5
            )
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1  # Parallel training
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                "Must be 'gbm' or 'rf'."
            )
        
        self.feature_names_ = None
        self._is_fitted = False
    
    def extract_features(
        self,
        R: np.ndarray,
        classifier_stats: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Extract features for each cell (u, i) in the matrix.
        
        Features are designed to capture:
        1. Cell-specific patterns (raw probability, certainty)
        2. Instance-level agreement (mean, std, range across classifiers)
        3. Classifier-level quality (optional performance metrics)
        
        Parameters
        ----------
        R : np.ndarray, shape (m, n)
            Probability matrix
        classifier_stats : dict, optional
            Dictionary mapping stat names to arrays of shape (m,)
            Example: {'accuracy': array([0.85, 0.78, ...]), 'auc': ...}
        
        Returns
        -------
        features : np.ndarray, shape (m*n, n_features)
            Feature matrix with one row per cell
        """
        m, n = R.shape
        features = []
        
        # === Cell-specific features ===
        # 1. Raw probability
        features.append(R.flatten())  # (m*n,)
        
        # 2. Distance from decision threshold (certainty)
        features.append(np.abs(R - 0.5).flatten())
        
        # === Instance-level features ===
        # Compute statistics across classifiers for each instance
        R_mean = np.mean(R, axis=0)  # (n,)
        R_std = np.std(R, axis=0)
        R_min = np.min(R, axis=0)
        R_max = np.max(R, axis=0)
        
        # Broadcast to (m*n,) by repeating m times per instance
        features.append(np.tile(R_mean, m))
        features.append(np.tile(R_std, m))
        features.append(np.tile(R_max - R_min, m))  # Range
        
        # === Classifier-level features (optional) ===
        if classifier_stats is not None:
            for stat_name, stat_values in classifier_stats.items():
                if stat_values.shape != (m,):
                    raise ValueError(
                        f"Classifier stat '{stat_name}' has shape "
                        f"{stat_values.shape}, expected ({m},)"
                    )
                # Broadcast to (m*n,) by repeating each classifier's stat n times
                features.append(np.repeat(stat_values, n))
        
        return np.column_stack(features)
    
    def fit(
        self,
        R: np.ndarray,
        labels: np.ndarray,
        labeled_idx: np.ndarray,
        classifier_stats: Optional[Dict[str, np.ndarray]] = None
    ) -> 'ReliabilityWeightModel':
        """
        Train reliability model on labeled data ONLY.
        
        The model learns to predict cell-level "correctness" defined as:
            target_ui = 1 - |r_ui - y_i|
        
        This provides m × |L| training examples (one per labeled cell).
        
        Parameters
        ----------
        R : np.ndarray, shape (m, n)
            Probability matrix from classifiers
        labels : np.ndarray, shape (n,)
            Ground truth labels with NaN for unlabeled instances
        labeled_idx : np.ndarray, shape (n,)
            Boolean mask indicating which instances are labeled
        classifier_stats : dict, optional
            Per-classifier performance metrics
        
        Returns
        -------
        self : ReliabilityWeightModel
            Fitted model
        
        Notes
        -----
        - Only labeled instances are used for training
        - The learned model can then predict weights for ALL cells
        - This is the key advantage: generalize to test set without pseudo-labels
        """
        m, n = R.shape
        
        # Validate inputs
        if labels.shape[0] != n:
            raise ValueError(f"labels shape {labels.shape} doesn't match R columns {n}")
        if labeled_idx.shape[0] != n:
            raise ValueError(f"labeled_idx shape {labeled_idx.shape} doesn't match R columns {n}")
        if np.sum(labeled_idx) == 0:
            raise ValueError("Must have at least one labeled instance")
        
        # Extract features for all cells
        features_all = self.extract_features(R, classifier_stats)
        
        # Create mask for labeled cells
        # For each labeled instance, we have m cells (one per classifier)
        # labeled_idx: (n,) boolean → repeat m times → (m*n,) boolean
        labeled_cell_mask = np.tile(labeled_idx, m)
        
        # Select features for labeled cells only
        features_labeled = features_all[labeled_cell_mask]
        
        # Compute continuous correctness targets
        # For labeled cell (u, i): target = 1 - |r_ui - y_i|
        R_labeled = R[:, labeled_idx]  # (m, |L|)
        y_labeled = labels[labeled_idx]  # (|L|,)
        
        # Broadcast y_labeled to match R_labeled shape
        y_broadcast = np.broadcast_to(y_labeled, R_labeled.shape)  # (m, |L|)
        
        # Compute targets: higher = more reliable
        targets = 1 - np.abs(R_labeled - y_broadcast)  # (m, |L|)
        targets_flat = targets.flatten()  # (m * |L|,)
        
        # Clip targets to [0, 1] (should already be, but ensure)
        targets_flat = np.clip(targets_flat, 0, 1)
        
        # Train model
        self.model.fit(features_labeled, targets_flat)
        
        # Store feature names for interpretability
        self.feature_names_ = [
            'prob', 'dist_threshold', 'mean', 'std', 'range'
        ]
        if classifier_stats:
            self.feature_names_.extend(classifier_stats.keys())
        
        self._is_fitted = True
        
        return self
    
    def predict(
        self,
        R: np.ndarray,
        classifier_stats: Optional[Dict[str, np.ndarray]] = None,
        clip_min: float = 0.1,
        clip_max: float = 1.0
    ) -> np.ndarray:
        """
        Predict reliability weights for all cells.
        
        No labels needed - works on both train and test data!
        
        Parameters
        ----------
        R : np.ndarray, shape (m, n)
            Probability matrix
        classifier_stats : dict, optional
            Same classifier statistics used during training
        clip_min : float, default=0.1
            Minimum weight (prevents zero weights in optimization)
        clip_max : float, default=1.0
            Maximum weight
        
        Returns
        -------
        W : np.ndarray, shape (m, n)
            Reliability weights in [clip_min, clip_max]
        
        Notes
        -----
        - Can be called on any data (train, validation, test)
        - No labels required for prediction
        - Weights are clipped to prevent numerical issues in ALS
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")
        
        # Extract features
        features = self.extract_features(R, classifier_stats)
        
        # Predict weights
        weights = self.model.predict(features)
        
        # Clip to valid range
        weights = np.clip(weights, clip_min, clip_max)
        
        # Reshape to matrix
        return weights.reshape(R.shape)
    
    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importances for tree-based models.
        
        Returns
        -------
        importances : dict or None
            Dictionary mapping feature names to importance scores
            Returns None if model doesn't support feature importances
        
        Notes
        -----
        Helps understand which features drive reliability predictions:
        - High importance for 'prob' → raw probability is key signal
        - High importance for 'std' → agreement matters
        - High importance for classifier stats → some classifiers more reliable
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.feature_names_,
                self.model.feature_importances_
            ))
        return None
    
    def get_params(self) -> Dict[str, any]:
        """Get model hyperparameters."""
        return {
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }
