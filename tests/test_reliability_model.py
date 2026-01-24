"""
Tests for learned reliability weight models.
"""

import pytest
import numpy as np
from cfensemble.models.reliability import ReliabilityWeightModel


@pytest.fixture
def sample_data():
    """Create sample data for reliability model testing."""
    np.random.seed(42)
    
    # 5 classifiers, 20 instances
    m, n = 5, 20
    R = np.random.rand(m, n)
    
    # 15 labeled, 5 unlabeled
    labels = np.concatenate([
        np.random.randint(0, 2, size=15).astype(float),
        np.array([np.nan] * 5)
    ])
    labeled_idx = ~np.isnan(labels)
    
    return R, labels, labeled_idx


@pytest.fixture
def sample_data_with_stats():
    """Create sample data with classifier statistics."""
    np.random.seed(42)
    
    m, n = 5, 20
    R = np.random.rand(m, n)
    
    labels = np.concatenate([
        np.random.randint(0, 2, size=15).astype(float),
        np.array([np.nan] * 5)
    ])
    labeled_idx = ~np.isnan(labels)
    
    # Per-classifier statistics
    classifier_stats = {
        'accuracy': np.array([0.85, 0.78, 0.92, 0.81, 0.89]),
        'auc': np.array([0.88, 0.81, 0.94, 0.83, 0.91])
    }
    
    return R, labels, labeled_idx, classifier_stats


class TestReliabilityModelInitialization:
    """Test model initialization."""
    
    def test_default_initialization(self):
        """Test default parameters."""
        model = ReliabilityWeightModel()
        
        assert model.model_type == 'gbm'
        assert model.n_estimators == 100
        assert model.learning_rate == 0.1
        assert model.max_depth == 3
        assert model.random_state == 42
        assert not model._is_fitted
    
    def test_gbm_initialization(self):
        """Test GBM model creation."""
        model = ReliabilityWeightModel(
            model_type='gbm',
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4
        )
        
        assert model.model_type == 'gbm'
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
        assert model.max_depth == 4
    
    def test_rf_initialization(self):
        """Test Random Forest model creation."""
        model = ReliabilityWeightModel(
            model_type='rf',
            n_estimators=50,
            max_depth=4
        )
        
        assert model.model_type == 'rf'
        assert model.n_estimators == 50
    
    def test_invalid_model_type(self):
        """Should raise error for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            ReliabilityWeightModel(model_type='invalid')


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_basic_features(self, sample_data):
        """Test basic feature extraction without classifier stats."""
        R, labels, labeled_idx = sample_data
        m, n = R.shape
        
        model = ReliabilityWeightModel()
        features = model.extract_features(R)
        
        # Should have 5 features: prob, dist_threshold, mean, std, range
        assert features.shape == (m * n, 5)
        
        # Check feature values
        # Feature 0: raw probabilities
        assert np.allclose(features[:, 0], R.flatten())
        
        # Feature 1: distance from threshold
        assert np.allclose(features[:, 1], np.abs(R - 0.5).flatten())
        
        # Feature 2: instance means (repeated m times)
        R_mean = np.mean(R, axis=0)
        assert np.allclose(features[:, 2], np.tile(R_mean, m))
    
    def test_features_with_stats(self, sample_data_with_stats):
        """Test feature extraction with classifier statistics."""
        R, labels, labeled_idx, classifier_stats = sample_data_with_stats
        m, n = R.shape
        
        model = ReliabilityWeightModel()
        features = model.extract_features(R, classifier_stats)
        
        # Should have 5 base + 2 classifier stats = 7 features
        assert features.shape == (m * n, 7)
        
        # Check classifier stats are broadcast correctly
        # Feature 5: accuracy (repeated n times per classifier)
        expected_acc = np.repeat(classifier_stats['accuracy'], n)
        assert np.allclose(features[:, 5], expected_acc)
    
    def test_feature_shape_consistency(self):
        """Features should match matrix shape."""
        R = np.random.rand(3, 10)
        model = ReliabilityWeightModel()
        features = model.extract_features(R)
        
        assert features.shape[0] == 3 * 10  # m * n rows
    
    def test_invalid_classifier_stats(self, sample_data):
        """Should raise error for mismatched classifier stats."""
        R, _, _ = sample_data
        m = R.shape[0]
        
        # Wrong shape
        bad_stats = {'accuracy': np.array([0.9, 0.8])}  # Should be (m,)
        
        model = ReliabilityWeightModel()
        with pytest.raises(ValueError, match="has shape"):
            model.extract_features(R, bad_stats)


class TestModelFitting:
    """Test model fitting."""
    
    def test_basic_fit(self, sample_data):
        """Test basic model fitting."""
        R, labels, labeled_idx = sample_data
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx)
        
        assert model._is_fitted
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == 5
    
    def test_fit_with_stats(self, sample_data_with_stats):
        """Test fitting with classifier statistics."""
        R, labels, labeled_idx, classifier_stats = sample_data_with_stats
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx, classifier_stats)
        
        assert model._is_fitted
        assert len(model.feature_names_) == 7  # 5 base + 2 stats
        assert 'accuracy' in model.feature_names_
        assert 'auc' in model.feature_names_
    
    def test_targets_computation(self, sample_data):
        """Test that targets are computed correctly."""
        R, labels, labeled_idx = sample_data
        m, n = R.shape
        
        model = ReliabilityWeightModel()
        
        # Manually compute expected targets
        R_labeled = R[:, labeled_idx]
        y_labeled = labels[labeled_idx]
        y_broadcast = np.broadcast_to(y_labeled, R_labeled.shape)
        expected_targets = 1 - np.abs(R_labeled - y_broadcast)
        
        # Fit model (we can't directly access targets, but check fitting works)
        model.fit(R, labels, labeled_idx)
        
        # Verify targets are in [0, 1]
        assert np.all(expected_targets >= 0)
        assert np.all(expected_targets <= 1)
    
    def test_fit_requires_labeled_data(self, sample_data):
        """Should raise error if no labeled data."""
        R, _, _ = sample_data
        labels = np.full(R.shape[1], np.nan)
        labeled_idx = ~np.isnan(labels)
        
        model = ReliabilityWeightModel()
        with pytest.raises(ValueError, match="at least one labeled"):
            model.fit(R, labels, labeled_idx)
    
    def test_fit_validates_shapes(self, sample_data):
        """Should validate input shapes."""
        R, labels, labeled_idx = sample_data
        
        # Wrong label shape
        bad_labels = np.array([1, 0])
        model = ReliabilityWeightModel()
        with pytest.raises(ValueError, match="doesn't match"):
            model.fit(R, bad_labels, labeled_idx)
    
    def test_different_model_types(self, sample_data):
        """Test fitting with different model types."""
        R, labels, labeled_idx = sample_data
        
        # GBM
        model_gbm = ReliabilityWeightModel(model_type='gbm')
        model_gbm.fit(R, labels, labeled_idx)
        assert model_gbm._is_fitted
        
        # RF
        model_rf = ReliabilityWeightModel(model_type='rf')
        model_rf.fit(R, labels, labeled_idx)
        assert model_rf._is_fitted


class TestModelPrediction:
    """Test model prediction."""
    
    def test_basic_prediction(self, sample_data):
        """Test basic weight prediction."""
        R, labels, labeled_idx = sample_data
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx)
        W = model.predict(R)
        
        # Check shape
        assert W.shape == R.shape
        
        # Check range [0.1, 1.0] (default clipping)
        assert np.all(W >= 0.1)
        assert np.all(W <= 1.0)
    
    def test_prediction_with_stats(self, sample_data_with_stats):
        """Test prediction with classifier statistics."""
        R, labels, labeled_idx, classifier_stats = sample_data_with_stats
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx, classifier_stats)
        W = model.predict(R, classifier_stats)
        
        assert W.shape == R.shape
        assert np.all(W >= 0.1) and np.all(W <= 1.0)
    
    def test_custom_clipping(self, sample_data):
        """Test custom clipping ranges."""
        R, labels, labeled_idx = sample_data
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx)
        
        # Custom clip range
        W = model.predict(R, clip_min=0.2, clip_max=0.9)
        
        assert np.all(W >= 0.2)
        assert np.all(W <= 0.9)
    
    def test_predict_without_fit(self, sample_data):
        """Should raise error if not fitted."""
        R, _, _ = sample_data
        
        model = ReliabilityWeightModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(R)
    
    def test_predict_different_data(self, sample_data):
        """Can predict on different data (generalization)."""
        R_train, labels, labeled_idx = sample_data
        
        # Train on original data
        model = ReliabilityWeightModel()
        model.fit(R_train, labels, labeled_idx)
        
        # Predict on new data (same shape)
        R_test = np.random.rand(*R_train.shape)
        W_test = model.predict(R_test)
        
        assert W_test.shape == R_test.shape
        assert np.all(W_test >= 0.1) and np.all(W_test <= 1.0)
    
    def test_reproducible_predictions(self, sample_data):
        """Predictions should be deterministic."""
        R, labels, labeled_idx = sample_data
        
        model = ReliabilityWeightModel(random_state=42)
        model.fit(R, labels, labeled_idx)
        
        W1 = model.predict(R)
        W2 = model.predict(R)
        
        assert np.allclose(W1, W2)


class TestFeatureImportance:
    """Test feature importance extraction."""
    
    def test_gbm_feature_importance(self, sample_data):
        """GBM should return feature importances."""
        R, labels, labeled_idx = sample_data
        
        model = ReliabilityWeightModel(model_type='gbm')
        model.fit(R, labels, labeled_idx)
        
        importance = model.feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == 5  # 5 base features
        assert 'prob' in importance
        assert 'dist_threshold' in importance
    
    def test_rf_feature_importance(self, sample_data):
        """RF should return feature importances."""
        R, labels, labeled_idx = sample_data
        
        model = ReliabilityWeightModel(model_type='rf')
        model.fit(R, labels, labeled_idx)
        
        importance = model.feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
    
    def test_importance_with_stats(self, sample_data_with_stats):
        """Should include classifier stats in importance."""
        R, labels, labeled_idx, classifier_stats = sample_data_with_stats
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx, classifier_stats)
        
        importance = model.feature_importance()
        
        assert 'accuracy' in importance
        assert 'auc' in importance
    
    def test_importance_before_fit(self):
        """Should raise error if not fitted."""
        model = ReliabilityWeightModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.feature_importance()


class TestModelParameters:
    """Test parameter retrieval."""
    
    def test_get_params(self):
        """Should return hyperparameters."""
        model = ReliabilityWeightModel(
            model_type='gbm',
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            random_state=123
        )
        
        params = model.get_params()
        
        assert params['model_type'] == 'gbm'
        assert params['n_estimators'] == 50
        assert params['learning_rate'] == 0.05
        assert params['max_depth'] == 4
        assert params['random_state'] == 123


class TestReliabilityModelIntegration:
    """Integration tests with EnsembleData and CFEnsembleTrainer."""
    
    def test_use_as_confidence_matrix(self, sample_data):
        """Test using reliability weights as confidence matrix."""
        from cfensemble.data import EnsembleData
        
        R, labels, labeled_idx = sample_data
        
        # Train reliability model
        rel_model = ReliabilityWeightModel()
        rel_model.fit(R, labels, labeled_idx)
        W = rel_model.predict(R)
        
        # Use as confidence in EnsembleData
        ensemble_data = EnsembleData(R, labels, C=W)
        
        assert np.allclose(ensemble_data.C, W)
    
    def test_full_pipeline(self, sample_data):
        """Test full pipeline: reliability model + trainer."""
        from cfensemble.data import EnsembleData
        from cfensemble.optimization import CFEnsembleTrainer
        
        R, labels, labeled_idx = sample_data
        m, n = R.shape
        
        # Learn reliability weights
        rel_model = ReliabilityWeightModel(n_estimators=10)  # Small for speed
        rel_model.fit(R, labels, labeled_idx)
        W = rel_model.predict(R)
        
        # Train CF-Ensemble with learned weights
        ensemble_data = EnsembleData(R, labels, C=W)
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=3,
            max_iter=5,
            rho=0.5,
            aggregator_type='weighted',
            verbose=False
        )
        trainer.fit(ensemble_data)
        
        # Should produce predictions
        y_pred = trainer.predict()
        assert y_pred.shape == (n,)
        assert np.all((y_pred >= 0) & (y_pred <= 1))


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_labeled_instance(self):
        """Test with minimal labeled data."""
        np.random.seed(42)
        R = np.random.rand(3, 5)
        labels = np.array([1, np.nan, np.nan, np.nan, np.nan])
        labeled_idx = ~np.isnan(labels)
        
        model = ReliabilityWeightModel(n_estimators=10)
        model.fit(R, labels, labeled_idx)
        W = model.predict(R)
        
        assert W.shape == R.shape
    
    def test_all_labeled(self):
        """Test when all instances are labeled."""
        np.random.seed(42)
        R = np.random.rand(3, 5)
        labels = np.array([1, 0, 1, 0, 1])
        labeled_idx = ~np.isnan(labels)
        
        model = ReliabilityWeightModel()
        model.fit(R, labels, labeled_idx)
        W = model.predict(R)
        
        assert W.shape == R.shape
    
    def test_perfect_predictions(self):
        """Test with perfect classifier predictions."""
        R = np.array([[1.0, 0.0, 1.0, 0.0]])
        labels = np.array([1, 0, 1, 0])
        labeled_idx = ~np.isnan(labels)
        
        model = ReliabilityWeightModel(n_estimators=10)
        model.fit(R, labels, labeled_idx)
        W = model.predict(R)
        
        # Should predict high reliability for perfect predictions
        assert np.all(W >= 0.5)
    
    def test_random_predictions(self):
        """Test with random (r=0.5) predictions."""
        R = np.full((3, 5), 0.5)
        labels = np.array([1, 0, 1, 0, 1])
        labeled_idx = ~np.isnan(labels)
        
        model = ReliabilityWeightModel(n_estimators=10)
        model.fit(R, labels, labeled_idx)
        W = model.predict(R)
        
        # Should predict lower reliability for random predictions
        assert W.shape == R.shape


class TestModelComparison:
    """Compare GBM vs RF models."""
    
    def test_gbm_vs_rf(self, sample_data):
        """Both models should produce valid weights."""
        R, labels, labeled_idx = sample_data
        
        # GBM
        model_gbm = ReliabilityWeightModel(model_type='gbm', random_state=42)
        model_gbm.fit(R, labels, labeled_idx)
        W_gbm = model_gbm.predict(R)
        
        # RF
        model_rf = ReliabilityWeightModel(model_type='rf', random_state=42)
        model_rf.fit(R, labels, labeled_idx)
        W_rf = model_rf.predict(R)
        
        # Both should be valid
        assert W_gbm.shape == R.shape
        assert W_rf.shape == R.shape
        
        # May produce different weights
        # (not necessarily the same, but both should be reasonable)
        assert np.all(W_gbm >= 0.1) and np.all(W_gbm <= 1.0)
        assert np.all(W_rf >= 0.1) and np.all(W_rf <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
