"""
Tests for confidence weighting strategies.
"""

import pytest
import numpy as np
from cfensemble.data.confidence import (
    ConfidenceStrategy,
    UniformConfidence,
    CertaintyConfidence,
    LabelAwareConfidence,
    CalibrationConfidence,
    AdaptiveConfidence,
    get_confidence_strategy
)


@pytest.fixture
def sample_data():
    """Create sample probability matrix and labels for testing."""
    # 5 classifiers, 10 instances
    np.random.seed(42)
    R = np.random.rand(5, 10)
    
    # 7 labeled, 3 unlabeled
    labels = np.array([1, 0, 1, 1, 0, 1, 0, np.nan, np.nan, np.nan])
    labeled_idx = ~np.isnan(labels)
    
    return R, labels, labeled_idx


class TestUniformConfidence:
    """Test uniform confidence strategy."""
    
    def test_all_ones(self, sample_data):
        """Should return all ones."""
        R, labels, _ = sample_data
        strategy = UniformConfidence()
        C = strategy.compute(R, labels)
        
        assert C.shape == R.shape
        assert np.allclose(C, 1.0)
    
    def test_without_labels(self, sample_data):
        """Should work without labels."""
        R, _, _ = sample_data
        strategy = UniformConfidence()
        C = strategy.compute(R)
        
        assert C.shape == R.shape
        assert np.allclose(C, 1.0)


class TestCertaintyConfidence:
    """Test certainty-based confidence strategy."""
    
    def test_distance_from_threshold(self, sample_data):
        """Should compute |r - 0.5|."""
        R, _, _ = sample_data
        strategy = CertaintyConfidence()
        C = strategy.compute(R)
        
        expected = np.abs(R - 0.5)
        assert np.allclose(C, expected)
    
    def test_extreme_values(self):
        """Test with extreme probabilities."""
        R = np.array([[0.0, 0.5, 1.0],
                      [0.1, 0.5, 0.9]])
        strategy = CertaintyConfidence()
        C = strategy.compute(R)
        
        expected = np.array([[0.5, 0.0, 0.5],
                            [0.4, 0.0, 0.4]])
        assert np.allclose(C, expected)
    
    def test_without_labels(self, sample_data):
        """Should work without labels."""
        R, _, _ = sample_data
        strategy = CertaintyConfidence()
        C = strategy.compute(R)
        
        assert C.shape == R.shape
        assert np.all(C >= 0) and np.all(C <= 0.5)


class TestLabelAwareConfidence:
    """Test label-aware confidence strategy."""
    
    def test_positive_labels(self):
        """For y=1, should return r."""
        R = np.array([[0.8, 0.2],
                      [0.9, 0.3]])
        labels = np.array([1, 0])
        
        strategy = LabelAwareConfidence()
        C = strategy.compute(R, labels)
        
        # For instance 0 (y=1): C[:, 0] = R[:, 0]
        assert np.allclose(C[:, 0], R[:, 0])
        
        # For instance 1 (y=0): C[:, 1] = 1 - R[:, 1]
        assert np.allclose(C[:, 1], 1 - R[:, 1])
    
    def test_unlabeled_fallback(self):
        """For unlabeled, should use certainty."""
        R = np.array([[0.8, 0.2],
                      [0.9, 0.3]])
        labels = np.array([1, np.nan])
        
        strategy = LabelAwareConfidence()
        C = strategy.compute(R, labels)
        
        # Instance 0 (labeled): use label-aware
        assert np.allclose(C[:, 0], R[:, 0])
        
        # Instance 1 (unlabeled): use certainty
        assert np.allclose(C[:, 1], np.abs(R[:, 1] - 0.5))
    
    def test_without_labels(self, sample_data):
        """Should fallback to certainty without labels."""
        R, _, _ = sample_data
        strategy = LabelAwareConfidence()
        C_no_labels = strategy.compute(R)
        
        certainty = CertaintyConfidence()
        C_certainty = certainty.compute(R)
        
        assert np.allclose(C_no_labels, C_certainty)
    
    def test_mixed_labels(self, sample_data):
        """Test with mix of labeled/unlabeled."""
        R, labels, labeled_idx = sample_data
        strategy = LabelAwareConfidence()
        C = strategy.compute(R, labels)
        
        # Check shape
        assert C.shape == R.shape
        
        # Check labeled instances use label-aware
        for i in np.where(labeled_idx)[0]:
            if labels[i] == 1:
                assert np.allclose(C[:, i], R[:, i])
            else:
                assert np.allclose(C[:, i], 1 - R[:, i])
        
        # Check unlabeled use certainty
        for i in np.where(~labeled_idx)[0]:
            assert np.allclose(C[:, i], np.abs(R[:, i] - 0.5))


class TestCalibrationConfidence:
    """Test calibration-based confidence strategy."""
    
    def test_good_classifier(self):
        """Good classifier should get high confidence."""
        # Perfect classifier
        R = np.array([[1.0, 0.0, 1.0],
                      [0.5, 0.5, 0.5]])  # Random classifier
        labels = np.array([1, 0, 1])
        
        strategy = CalibrationConfidence(floor=0.1)
        C = strategy.compute(R, labels)
        
        # Classifier 0: Brier = 0, calibration = 1
        assert np.allclose(C[0, :], 1.0)
        
        # Classifier 1: Brier = 0.25, calibration = 0.75
        assert np.allclose(C[1, :], 0.75)
    
    def test_bad_classifier(self):
        """Bad classifier should get low confidence (floor)."""
        # Anti-correlated classifier
        R = np.array([[0.0, 1.0, 0.0]])
        labels = np.array([1, 0, 1])
        
        strategy = CalibrationConfidence(floor=0.1)
        C = strategy.compute(R, labels)
        
        # Brier = 1.0, calibration = 0 → floor
        assert np.allclose(C[0, :], 0.1)
    
    def test_floor_parameter(self):
        """Test different floor values."""
        R = np.array([[0.0, 0.0]])
        labels = np.array([1, 1])
        
        strategy = CalibrationConfidence(floor=0.2)
        C = strategy.compute(R, labels)
        
        # Very bad classifier → should hit floor
        assert np.allclose(C[0, :], 0.2)
    
    def test_without_labels_fallback(self, sample_data):
        """Without labels, should return uniform."""
        R, _, _ = sample_data
        strategy = CalibrationConfidence()
        C = strategy.compute(R)
        
        assert np.allclose(C, 1.0)
    
    def test_per_classifier_weights(self):
        """Test that weights are per-classifier, not per-cell."""
        R = np.array([[0.9, 0.1],
                      [0.1, 0.9]])
        labels = np.array([1, 0])
        
        strategy = CalibrationConfidence()
        C = strategy.compute(R, labels)
        
        # Each row should have constant values
        assert np.allclose(C[0, 0], C[0, 1])
        assert np.allclose(C[1, 0], C[1, 1])


class TestAdaptiveConfidence:
    """Test adaptive confidence strategy."""
    
    def test_weights_sum_to_one(self):
        """Weights should sum to 1."""
        strategy = AdaptiveConfidence(alpha=0.5, beta=0.3, gamma=0.2)
        assert np.isclose(strategy.alpha + strategy.beta + strategy.gamma, 1.0)
    
    def test_invalid_weights(self):
        """Should raise error if weights don't sum to 1."""
        with pytest.raises(ValueError):
            AdaptiveConfidence(alpha=0.5, beta=0.5, gamma=0.5)
    
    def test_combines_strategies(self, sample_data):
        """Should combine multiple strategies."""
        R, labels, _ = sample_data
        
        strategy = AdaptiveConfidence(alpha=0.5, beta=0.3, gamma=0.2)
        C = strategy.compute(R, labels)
        
        # Compute individual components
        certainty = CertaintyConfidence()
        calibration = CalibrationConfidence()
        
        C_certainty = certainty.compute(R, labels)
        C_calibration = calibration.compute(R, labels)
        
        # Agreement component
        R_std = np.std(R, axis=0)
        R_std_normalized = R_std / (np.max(R_std) + 1e-8)
        C_agreement = 1 - R_std_normalized
        C_agreement = np.broadcast_to(C_agreement, R.shape)
        
        # Combined
        expected = (0.5 * C_certainty + 
                   0.3 * C_calibration + 
                   0.2 * C_agreement)
        
        assert np.allclose(C, expected)
    
    def test_different_weight_combinations(self, sample_data):
        """Test different weight combinations."""
        R, labels, _ = sample_data
        
        # Certainty-heavy
        strategy1 = AdaptiveConfidence(alpha=0.7, beta=0.2, gamma=0.1)
        C1 = strategy1.compute(R, labels)
        
        # Calibration-heavy
        strategy2 = AdaptiveConfidence(alpha=0.2, beta=0.7, gamma=0.1)
        C2 = strategy2.compute(R, labels)
        
        # Should produce different results
        assert not np.allclose(C1, C2)


class TestConfidenceFactory:
    """Test get_confidence_strategy factory function."""
    
    def test_all_strategies(self):
        """Test that all strategies can be created."""
        strategies = ['uniform', 'certainty', 'label_aware', 
                     'calibration', 'adaptive']
        
        for name in strategies:
            strategy = get_confidence_strategy(name)
            assert isinstance(strategy, ConfidenceStrategy)
    
    def test_with_parameters(self):
        """Test creating strategies with parameters."""
        strategy = get_confidence_strategy('calibration', floor=0.2)
        assert isinstance(strategy, CalibrationConfidence)
        assert strategy.floor == 0.2
        
        strategy = get_confidence_strategy('adaptive', 
                                          alpha=0.5, beta=0.3, gamma=0.2)
        assert isinstance(strategy, AdaptiveConfidence)
        assert strategy.alpha == 0.5
    
    def test_invalid_strategy(self):
        """Should raise error for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_confidence_strategy('nonexistent')


class TestConfidenceProperties:
    """Test general properties of confidence strategies."""
    
    def test_output_shape(self, sample_data):
        """All strategies should preserve shape."""
        R, labels, _ = sample_data
        
        strategies = [
            UniformConfidence(),
            CertaintyConfidence(),
            LabelAwareConfidence(),
            CalibrationConfidence(),
            AdaptiveConfidence()
        ]
        
        for strategy in strategies:
            C = strategy.compute(R, labels)
            assert C.shape == R.shape
    
    def test_non_negative(self, sample_data):
        """All confidences should be non-negative."""
        R, labels, _ = sample_data
        
        strategies = [
            UniformConfidence(),
            CertaintyConfidence(),
            LabelAwareConfidence(),
            CalibrationConfidence(),
            AdaptiveConfidence()
        ]
        
        for strategy in strategies:
            C = strategy.compute(R, labels)
            assert np.all(C >= 0)
    
    def test_deterministic(self, sample_data):
        """Should produce same results on repeated calls."""
        R, labels, _ = sample_data
        
        strategy = AdaptiveConfidence()
        C1 = strategy.compute(R, labels)
        C2 = strategy.compute(R, labels)
        
        assert np.allclose(C1, C2)


class TestEdgeCases:
    """Test edge cases for confidence strategies."""
    
    def test_single_classifier(self):
        """Test with single classifier."""
        R = np.array([[0.7, 0.3, 0.9]])
        labels = np.array([1, 0, 1])
        
        strategy = CertaintyConfidence()
        C = strategy.compute(R, labels)
        
        assert C.shape == (1, 3)
    
    def test_single_instance(self):
        """Test with single instance."""
        R = np.array([[0.7], [0.3], [0.9]])
        labels = np.array([1])
        
        strategy = LabelAwareConfidence()
        C = strategy.compute(R, labels)
        
        assert C.shape == (3, 1)
    
    def test_all_labeled(self):
        """Test when all instances are labeled."""
        R = np.random.rand(3, 4)
        labels = np.array([1, 0, 1, 0])
        
        strategy = LabelAwareConfidence()
        C = strategy.compute(R, labels)
        
        assert C.shape == R.shape
    
    def test_all_unlabeled(self):
        """Test when all instances are unlabeled."""
        R = np.random.rand(3, 4)
        labels = np.array([np.nan, np.nan, np.nan, np.nan])
        
        strategy = LabelAwareConfidence()
        C = strategy.compute(R, labels)
        
        # Should fallback to certainty
        certainty = CertaintyConfidence()
        C_expected = certainty.compute(R)
        assert np.allclose(C, C_expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
