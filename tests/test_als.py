"""
Tests for Alternating Least Squares (ALS) update functions.
"""

import numpy as np
import pytest
from src.cfensemble.optimization.als import (
    update_classifier_factors,
    update_instance_factors,
    compute_reconstruction_error
)


class TestALSUpdates:
    """Test ALS update functions."""
    
    def test_update_classifier_factors_shape(self):
        """Test that X has correct shape after update."""
        d, m, n = 10, 5, 20
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        X = update_classifier_factors(Y, R, C, lambda_reg)
        
        assert X.shape == (d, m), f"Expected shape ({d}, {m}), got {X.shape}"
    
    def test_update_instance_factors_shape(self):
        """Test that Y has correct shape after update."""
        d, m, n = 10, 5, 20
        X = np.random.randn(d, m)
        R = np.random.rand(m, n)
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        Y = update_instance_factors(X, R, C, lambda_reg)
        
        assert Y.shape == (d, n), f"Expected shape ({d}, {n}), got {Y.shape}"
    
    def test_als_reduces_reconstruction_error(self):
        """Test that ALS updates reduce reconstruction error."""
        np.random.seed(42)
        d, m, n = 5, 10, 50
        
        # Generate synthetic data
        X_true = np.random.randn(d, m)
        Y_true = np.random.randn(d, n)
        R = X_true.T @ Y_true + np.random.randn(m, n) * 0.1
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        # Initialize randomly
        X = np.random.randn(d, m) * 0.01
        Y = np.random.randn(d, n) * 0.01
        
        # Initial error
        initial_error = compute_reconstruction_error(X, Y, R, C)
        
        # Run a few ALS iterations
        for _ in range(10):
            X = update_classifier_factors(Y, R, C, lambda_reg)
            Y = update_instance_factors(X, R, C, lambda_reg)
        
        # Final error
        final_error = compute_reconstruction_error(X, Y, R, C)
        
        assert final_error < initial_error, \
            f"Error should decrease: {initial_error:.4f} -> {final_error:.4f}"
    
    def test_als_convergence_on_perfect_data(self):
        """Test ALS convergence on noise-free data."""
        np.random.seed(42)
        d, m, n = 5, 10, 30
        
        # Generate perfect low-rank data
        X_true = np.random.randn(d, m)
        Y_true = np.random.randn(d, n)
        R = X_true.T @ Y_true  # No noise
        C = np.ones((m, n))
        lambda_reg = 0.001  # Small regularization
        
        # Initialize randomly
        X = np.random.randn(d, m) * 0.1
        Y = np.random.randn(d, n) * 0.1
        
        # Run ALS
        errors = []
        for _ in range(50):
            X = update_classifier_factors(Y, R, C, lambda_reg)
            Y = update_instance_factors(X, R, C, lambda_reg)
            error = compute_reconstruction_error(X, Y, R, C)
            errors.append(error)
        
        # Should converge to very small error
        final_rmse = np.sqrt(errors[-1] / (m * n))
        assert final_rmse < 0.01, f"Should converge to near-zero error, got RMSE={final_rmse:.4f}"
        
        # Error should be monotonically decreasing (approximately)
        assert errors[-1] < errors[0], "Error should decrease over iterations"
    
    def test_confidence_weighting_effect(self):
        """Test that high confidence predictions are fit better."""
        np.random.seed(42)
        d, m, n = 5, 3, 20
        
        # Create data
        X = np.random.randn(d, m)
        Y = np.random.randn(d, n)
        R = X.T @ Y
        
        # High confidence on first half, low on second half
        C = np.ones((m, n))
        C[:, :n//2] = 10.0  # High confidence
        C[:, n//2:] = 0.1   # Low confidence
        
        lambda_reg = 0.01
        
        # Run ALS
        for _ in range(20):
            X = update_classifier_factors(Y, R, C, lambda_reg)
            Y = update_instance_factors(X, R, C, lambda_reg)
        
        # Compute reconstruction
        R_hat = X.T @ Y
        
        # High confidence region should have smaller errors
        error_high = np.mean((R[:, :n//2] - R_hat[:, :n//2])**2)
        error_low = np.mean((R[:, n//2:] - R_hat[:, n//2:])**2)
        
        assert error_high < error_low, \
            "High confidence region should be fit better"
    
    def test_regularization_effect(self):
        """Test that regularization prevents overfitting."""
        np.random.seed(42)
        d, m, n = 10, 5, 20
        
        # Generate noisy data
        X_true = np.random.randn(d, m) * 0.1
        Y_true = np.random.randn(d, n) * 0.1
        R = X_true.T @ Y_true + np.random.randn(m, n) * 0.5  # High noise
        C = np.ones((m, n))
        
        # Train with different regularization
        lambda_small = 0.001
        lambda_large = 1.0
        
        # Small regularization
        X_small = np.random.randn(d, m) * 0.01
        Y_small = np.random.randn(d, n) * 0.01
        for _ in range(30):
            X_small = update_classifier_factors(Y_small, R, C, lambda_small)
            Y_small = update_instance_factors(X_small, R, C, lambda_small)
        
        # Large regularization
        X_large = np.random.randn(d, m) * 0.01
        Y_large = np.random.randn(d, n) * 0.01
        for _ in range(30):
            X_large = update_classifier_factors(Y_large, R, C, lambda_large)
            Y_large = update_instance_factors(X_large, R, C, lambda_large)
        
        # Large regularization should produce smaller factor norms
        norm_small = np.linalg.norm(X_small) + np.linalg.norm(Y_small)
        norm_large = np.linalg.norm(X_large) + np.linalg.norm(Y_large)
        
        assert norm_large < norm_small, \
            "Larger regularization should produce smaller norms"
    
    def test_single_classifier(self):
        """Test ALS with single classifier (edge case)."""
        d, m, n = 5, 1, 10
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        X = update_classifier_factors(Y, R, C, lambda_reg)
        
        assert X.shape == (d, m)
        assert not np.any(np.isnan(X)), "Should not produce NaN"
    
    def test_single_instance(self):
        """Test ALS with single instance (edge case)."""
        d, m, n = 5, 10, 1
        X = np.random.randn(d, m)
        R = np.random.rand(m, n)
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        Y = update_instance_factors(X, R, C, lambda_reg)
        
        assert Y.shape == (d, n)
        assert not np.any(np.isnan(Y)), "Should not produce NaN"


class TestReconstructionError:
    """Test reconstruction error computation."""
    
    def test_perfect_reconstruction(self):
        """Test error is zero for perfect reconstruction."""
        d, m, n = 5, 10, 20
        X = np.random.randn(d, m)
        Y = np.random.randn(d, n)
        R = X.T @ Y  # Perfect reconstruction
        C = np.ones((m, n))
        
        error = compute_reconstruction_error(X, Y, R, C)
        
        assert error < 1e-10, f"Perfect reconstruction should have zero error, got {error}"
    
    def test_error_is_positive(self):
        """Test that error is always non-negative."""
        d, m, n = 5, 10, 20
        X = np.random.randn(d, m)
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)  # Random, won't match X.T @ Y
        C = np.ones((m, n))
        
        error = compute_reconstruction_error(X, Y, R, C)
        
        assert error >= 0, "Error should be non-negative"
    
    def test_confidence_weighting_in_error(self):
        """Test that confidence weights affect error magnitude."""
        d, m, n = 5, 10, 20
        X = np.random.randn(d, m)
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)
        
        # Uniform confidence
        C_uniform = np.ones((m, n))
        error_uniform = compute_reconstruction_error(X, Y, R, C_uniform)
        
        # High confidence everywhere
        C_high = np.ones((m, n)) * 10.0
        error_high = compute_reconstruction_error(X, Y, R, C_high)
        
        # High confidence should increase error magnitude
        assert error_high > error_uniform, \
            "Higher confidence should increase weighted error"
    
    def test_error_with_zero_confidence(self):
        """Test error computation with zero confidence weights."""
        d, m, n = 5, 10, 20
        X = np.random.randn(d, m)
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)
        C = np.zeros((m, n))  # Zero confidence everywhere
        
        error = compute_reconstruction_error(X, Y, R, C)
        
        assert error == 0, "Zero confidence should give zero error"


class TestALSNumericalStability:
    """Test numerical stability of ALS updates."""
    
    def test_no_nans_with_large_values(self):
        """Test that large input values don't produce NaN."""
        d, m, n = 5, 10, 20
        Y = np.random.randn(d, n) * 100  # Large values
        R = np.random.rand(m, n) * 10
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        X = update_classifier_factors(Y, R, C, lambda_reg)
        
        assert not np.any(np.isnan(X)), "Should not produce NaN with large values"
        assert not np.any(np.isinf(X)), "Should not produce Inf with large values"
    
    def test_no_nans_with_small_regularization(self):
        """Test stability with very small regularization."""
        d, m, n = 5, 10, 20
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)
        C = np.ones((m, n))
        lambda_reg = 1e-10  # Very small
        
        X = update_classifier_factors(Y, R, C, lambda_reg)
        
        assert not np.any(np.isnan(X)), "Should not produce NaN with small lambda"
    
    def test_conditioning_with_zero_regularization(self):
        """Test that zero regularization can cause numerical issues."""
        d, m, n = 5, 10, 20
        Y = np.random.randn(d, n)
        R = np.random.rand(m, n)
        C = np.ones((m, n))
        lambda_reg = 0.0  # No regularization
        
        # This might produce warnings or errors depending on data
        # We just check it doesn't crash
        try:
            X = update_classifier_factors(Y, R, C, lambda_reg)
            # If it works, check for numerical issues
            has_issues = np.any(np.isnan(X)) or np.any(np.isinf(X))
            # It's OK if it has issues, just documenting the behavior
        except np.linalg.LinAlgError:
            # Singular matrix is expected with zero regularization
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
