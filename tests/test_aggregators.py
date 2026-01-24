"""
Tests for aggregator classes.

Run with: pytest tests/test_aggregators.py -v
"""

import numpy as np
import pytest
from src.cfensemble.ensemble import MeanAggregator, WeightedAggregator


class TestMeanAggregator:
    """Test mean aggregator."""
    
    def test_predict(self):
        """Test mean aggregation."""
        agg = MeanAggregator()
        
        r_hat = np.array([[0.2, 0.8],
                          [0.4, 0.6],
                          [0.6, 0.4]])
        
        predictions = agg.predict(r_hat)
        
        expected = np.array([0.4, 0.6])
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_no_update(self):
        """Test that update doesn't change anything."""
        agg = MeanAggregator()
        
        # Update should do nothing
        X = np.random.randn(20, 10)
        Y = np.random.randn(20, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        labeled_idx = np.arange(100) < 50
        
        agg.update(X, Y, labeled_idx, labels, lr=0.01)
        
        # No parameters to check, just verify no error


class TestWeightedAggregator:
    """Test weighted aggregator."""
    
    def test_initialization_uniform(self):
        """Test uniform initialization."""
        agg = WeightedAggregator(n_classifiers=5, init_uniform=True)
        
        expected_w = np.ones(5) / 5
        np.testing.assert_array_almost_equal(agg.w, expected_w)
        assert agg.b == 0.0
    
    def test_initialization_random(self):
        """Test random initialization."""
        agg = WeightedAggregator(n_classifiers=5, init_uniform=False)
        
        # Weights should be small but not uniform
        assert not np.allclose(agg.w, np.ones(5) / 5)
        assert np.all(np.abs(agg.w) < 1)
    
    def test_predict(self):
        """Test prediction."""
        agg = WeightedAggregator(n_classifiers=3, init_uniform=True)
        agg.w = np.array([0.5, 0.3, 0.2])
        agg.b = 0.0
        
        r_hat = np.array([[0.8, 0.2],
                          [0.6, 0.4],
                          [0.7, 0.3]])
        
        predictions = agg.predict(r_hat)
        
        # logits = [0.5*0.8 + 0.3*0.6 + 0.2*0.7, 0.5*0.2 + 0.3*0.4 + 0.2*0.3]
        logits = np.array([0.72, 0.28])
        expected = 1 / (1 + np.exp(-logits))
        
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_update(self):
        """Test weight update."""
        np.random.seed(42)
        agg = WeightedAggregator(n_classifiers=10, init_uniform=True)
        
        w_before = agg.w.copy()
        b_before = agg.b
        
        X = np.random.randn(20, 10) * 0.1
        Y = np.random.randn(20, 100) * 0.1
        labels = np.random.randint(0, 2, 100).astype(float)
        labeled_idx = np.arange(100) < 50
        
        agg.update(X, Y, labeled_idx, labels, lr=0.01)
        
        # Weights should have changed
        assert not np.allclose(agg.w, w_before)
        assert agg.b != b_before
    
    def test_get_set_params(self):
        """Test parameter getting and setting."""
        agg = WeightedAggregator(n_classifiers=5)
        agg.w = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        agg.b = 0.5
        
        params = agg.get_params()
        
        agg2 = WeightedAggregator(n_classifiers=5)
        agg2.set_params(params)
        
        np.testing.assert_array_equal(agg2.w, agg.w)
        assert agg2.b == agg.b
    
    def test_get_normalized_weights(self):
        """Test normalized weight retrieval."""
        agg = WeightedAggregator(n_classifiers=4)
        agg.w = np.array([1.0, 2.0, -1.0, 0.0])
        
        normalized = agg.get_normalized_weights()
        
        # Should sum to 1 and use absolute values
        assert np.sum(normalized) == pytest.approx(1.0)
        assert np.all(normalized >= 0)
        
        expected = np.array([1, 2, 1, 0]) / 4
        np.testing.assert_array_almost_equal(normalized, expected)


class TestAggregatorEdgeCases:
    """Test edge cases for aggregators."""
    
    def test_single_classifier(self):
        """Test with single classifier."""
        agg_mean = MeanAggregator()
        agg_weighted = WeightedAggregator(n_classifiers=1)
        
        r_hat = np.array([[0.7, 0.3]])
        
        pred_mean = agg_mean.predict(r_hat)
        pred_weighted = agg_weighted.predict(r_hat)
        
        # With 1 classifier, should be close to input (after sigmoid)
        assert pred_mean.shape == (2,)
        assert pred_weighted.shape == (2,)
    
    def test_extreme_probabilities(self):
        """Test with probabilities close to 0 and 1."""
        agg = MeanAggregator()
        
        r_hat = np.array([[0.0001, 0.9999],
                          [0.0002, 0.9998],
                          [0.0003, 0.9997]])
        
        predictions = agg.predict(r_hat)
        
        # Should handle extreme values
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert predictions[0] < 0.01
        assert predictions[1] > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
