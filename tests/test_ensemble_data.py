"""
Tests for EnsembleData class.

Run with: pytest tests/test_ensemble_data.py -v
"""

import numpy as np
import pytest
from src.cfensemble.data import EnsembleData


class TestEnsembleDataBasic:
    """Test basic functionality of EnsembleData."""
    
    def test_initialization(self):
        """Test basic initialization."""
        m, n = 10, 1000
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        labels[500:] = np.nan
        
        data = EnsembleData(R, labels)
        
        assert data.n_classifiers == m
        assert data.n_instances == n
        assert data.n_labeled == 500
        assert data.n_unlabeled == 500
        assert data.R.shape == (m, n)
        assert data.C.shape == (m, n)
    
    def test_confidence_computation(self):
        """Test default confidence computation."""
        R = np.array([[0.1, 0.5, 0.9],
                      [0.3, 0.5, 0.7]])
        labels = np.array([0.0, 1.0, np.nan])
        
        data = EnsembleData(R, labels)
        
        # Certainty-based: |r - 0.5|
        expected_C = np.abs(R - 0.5)
        np.testing.assert_array_almost_equal(data.C, expected_C)
    
    def test_labeled_unlabeled_masks(self):
        """Test labeled/unlabeled masks are correct."""
        R = np.random.rand(5, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        labels[60:] = np.nan
        
        data = EnsembleData(R, labels)
        
        assert np.sum(data.labeled_idx) == 60
        assert np.sum(data.unlabeled_idx) == 40
        assert np.all(~np.isnan(labels[data.labeled_idx]))
        assert np.all(np.isnan(labels[data.unlabeled_idx]))
    
    def test_get_labeled_data(self):
        """Test retrieval of labeled data."""
        R = np.random.rand(5, 10)
        labels = np.array([0, 1, 0, 1, 0, np.nan, np.nan, np.nan, 1, 0], dtype=float)
        # Labeled: indices 0,1,2,3,4,8,9 = 7 total
        # Unlabeled: indices 5,6,7 = 3 total
        
        data = EnsembleData(R, labels)
        R_labeled, y_labeled, C_labeled = data.get_labeled_data()
        
        assert R_labeled.shape == (5, 7)  # 7 labeled instances
        assert y_labeled.shape == (7,)
        assert C_labeled.shape == (5, 7)
        assert not np.any(np.isnan(y_labeled))
    
    def test_get_unlabeled_data(self):
        """Test retrieval of unlabeled data."""
        R = np.random.rand(5, 10)
        labels = np.array([0, 1, 0, 1, 0, np.nan, np.nan, np.nan, 1, 0], dtype=float)
        # Unlabeled: indices 5,6,7 = 3 total
        
        data = EnsembleData(R, labels)
        R_unlabeled, C_unlabeled = data.get_unlabeled_data()
        
        assert R_unlabeled.shape == (5, 3)  # 3 unlabeled instances
        assert C_unlabeled.shape == (5, 3)


class TestEnsembleDataValidation:
    """Test input validation."""
    
    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 50).astype(float)  # Wrong size
        
        with pytest.raises(ValueError, match="must match"):
            EnsembleData(R, labels)
    
    def test_invalid_probabilities(self):
        """Test that probabilities outside [0,1] raise error."""
        R = np.random.rand(10, 100) * 2  # Values > 1
        labels = np.random.randint(0, 2, 100).astype(float)
        
        with pytest.raises(ValueError, match="probabilities in"):
            EnsembleData(R, labels)
    
    def test_no_labeled_data(self):
        """Test that all-unlabeled data is allowed (for pure reconstruction)."""
        R = np.random.rand(10, 100)
        labels = np.full(100, np.nan)
        
        # Should not raise error - pure reconstruction is valid
        data = EnsembleData(R, labels)
        assert data.n_labeled == 0
        assert data.n_unlabeled == 100
    
    def test_invalid_labels(self):
        """Test that non-binary labels raise error."""
        R = np.random.rand(10, 100)
        labels = np.random.rand(100)  # Continuous values
        
        with pytest.raises(ValueError, match="must be 0 or 1"):
            EnsembleData(R, labels)
    
    def test_confidence_dimension_mismatch(self):
        """Test that confidence with wrong shape raises error."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        C = np.random.rand(5, 100)  # Wrong number of classifiers
        
        with pytest.raises(ValueError, match="must match"):
            EnsembleData(R, labels, C=C)


class TestEnsembleDataSplit:
    """Test train/validation splitting."""
    
    def test_split_preserves_unlabeled(self):
        """Test that splitting preserves unlabeled data in both sets."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        labels[60:] = np.nan  # 40 unlabeled
        
        data = EnsembleData(R, labels)
        train_data, val_data = data.split_labeled_data(train_fraction=0.8, random_state=42)
        
        # Split masks opposite sets, so counts change
        # Train gets: 48 labeled + (40 original unlabeled + 12 masked val) = 48 labeled + 52 unlabeled
        # Val gets: 12 labeled + (40 original unlabeled + 48 masked train) = 12 labeled + 88 unlabeled
        # This is correct behavior for transductive learning
        
        # Train should have 80% of 60 labeled = 48
        assert train_data.n_labeled == pytest.approx(48, abs=2)
        assert train_data.n_unlabeled == pytest.approx(52, abs=2)
        
        # Val should have remaining ~12
        assert val_data.n_labeled == pytest.approx(12, abs=2)
        assert val_data.n_unlabeled == pytest.approx(88, abs=2)
        
        # Total instances should be preserved
        assert train_data.n_instances == 100
        assert val_data.n_instances == 100
    
    def test_split_reproducibility(self):
        """Test that same random_state gives same split."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        labels[60:] = np.nan
        
        data = EnsembleData(R, labels)
        
        train1, val1 = data.split_labeled_data(random_state=42)
        train2, val2 = data.split_labeled_data(random_state=42)
        
        np.testing.assert_array_equal(train1.labeled_idx, train2.labeled_idx)
        np.testing.assert_array_equal(val1.labeled_idx, val2.labeled_idx)


class TestEnsembleDataUtilities:
    """Test utility methods."""
    
    def test_update_confidence(self):
        """Test confidence matrix update."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        
        data = EnsembleData(R, labels)
        C_old = data.C.copy()
        
        C_new = np.ones_like(R)
        data.update_confidence(C_new)
        
        assert not np.array_equal(data.C, C_old)
        np.testing.assert_array_equal(data.C, C_new)
    
    def test_repr(self):
        """Test string representation."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        labels[60:] = np.nan
        
        data = EnsembleData(R, labels)
        repr_str = repr(data)
        
        assert "EnsembleData" in repr_str
        assert "10" in repr_str  # n_classifiers
        assert "100" in repr_str  # n_instances
    
    def test_summary(self):
        """Test summary method."""
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        labels[60:] = np.nan
        
        data = EnsembleData(R, labels)
        summary = data.summary()
        
        assert "Number of classifiers" in summary
        assert "Labeled" in summary
        assert "Unlabeled" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
