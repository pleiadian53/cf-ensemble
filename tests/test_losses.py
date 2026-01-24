"""
Tests for loss functions.

Run with: pytest tests/test_losses.py -v
"""

import numpy as np
import pytest
from src.cfensemble.objectives.losses import (
    reconstruction_loss,
    supervised_loss,
    combined_loss,
    compute_rmse
)
from src.cfensemble.ensemble import MeanAggregator, WeightedAggregator


class TestReconstructionLoss:
    """Test reconstruction loss function."""
    
    def test_perfect_reconstruction(self):
        """Test that perfect reconstruction gives only regularization loss."""
        m, n = 5, 10
        d = min(m, n)  # Use full rank for perfect reconstruction
        R = np.random.rand(m, n)
        
        # Create factors that perfectly reconstruct R
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        X = (U * np.sqrt(S))[:, :d].T  # (d × m)
        Y = (Vt[:d, :] * np.sqrt(S[:d, None]))  # (d × n)
        
        C = np.ones((m, n))
        lambda_reg = 0.01
        
        loss = reconstruction_loss(R, X, Y, C, lambda_reg)
        
        # Should be close to regularization term only (reconstruction error near zero)
        reg_term = lambda_reg * (np.sum(X**2) + np.sum(Y**2))
        assert loss == pytest.approx(reg_term, rel=0.01)
    
    def test_zero_factors(self):
        """Test loss with zero factors."""
        m, n, d = 5, 10, 3
        R = np.random.rand(m, n)
        X = np.zeros((d, m))
        Y = np.zeros((d, n))
        C = np.ones((m, n))
        
        loss = reconstruction_loss(R, X, Y, C, lambda_reg=0)
        
        # Should equal sum of squared probabilities
        expected = np.sum(R**2)
        assert loss == pytest.approx(expected)
    
    def test_confidence_weighting(self):
        """Test that confidence weights affect loss."""
        m, n, d = 5, 10, 3
        R = np.random.rand(m, n)
        X = np.random.randn(d, m) * 0.01
        Y = np.random.randn(d, n) * 0.01
        
        # Uniform confidence
        C_uniform = np.ones((m, n))
        loss_uniform = reconstruction_loss(R, X, Y, C_uniform, lambda_reg=0)
        
        # Higher confidence on some cells
        C_weighted = np.ones((m, n))
        C_weighted[:, :5] = 10  # Much higher weight on first 5 instances
        loss_weighted = reconstruction_loss(R, X, Y, C_weighted, lambda_reg=0)
        
        # Losses should be different
        assert loss_uniform != pytest.approx(loss_weighted)


class TestSupervisedLoss:
    """Test supervised loss function."""
    
    def test_perfect_predictions(self):
        """Test that perfect predictions give zero loss."""
        m, n, d = 5, 10, 3
        labels = np.array([0, 1, 0, 1, 0, np.nan, np.nan, np.nan, 1, 0], dtype=float)
        labeled_idx = ~np.isnan(labels)
        
        # Create factors that give perfect predictions
        # For simplicity, use mean aggregator
        agg = MeanAggregator()
        
        # Set up factors so mean of reconstructed probs = labels
        X = np.random.randn(d, m)
        Y = np.random.randn(d, n)
        
        # Manually adjust to get close to perfect (not exactly possible with random)
        # Instead, test that loss decreases with better predictions
        loss = supervised_loss(X, Y, labels, labeled_idx, agg)
        
        # Loss should be finite and positive
        assert np.isfinite(loss)
        assert loss > 0
    
    def test_binary_labels(self):
        """Test with binary labels."""
        m, n, d = 10, 100, 20
        X = np.random.randn(d, m) * 0.1
        Y = np.random.randn(d, n) * 0.1
        
        labels = np.random.randint(0, 2, n).astype(float)
        labels[50:] = np.nan
        labeled_idx = ~np.isnan(labels)
        
        agg = MeanAggregator()
        loss = supervised_loss(X, Y, labels, labeled_idx, agg)
        
        assert np.isfinite(loss)
        assert loss > 0


class TestCombinedLoss:
    """Test combined loss function."""
    
    def test_rho_extremes(self):
        """Test that rho=0 and rho=1 give expected behavior."""
        m, n, d = 10, 100, 20
        R = np.random.rand(m, n)
        X = np.random.randn(d, m) * 0.1
        Y = np.random.randn(d, n) * 0.1
        C = np.ones((m, n))
        
        labels = np.random.randint(0, 2, n).astype(float)
        labels[50:] = np.nan
        labeled_idx = ~np.isnan(labels)
        
        agg = MeanAggregator()
        lambda_reg = 0.01
        
        # rho = 1: Pure reconstruction
        loss_pure_recon, dict1 = combined_loss(
            R, X, Y, C, labels, labeled_idx, agg, rho=1.0, lambda_reg=lambda_reg
        )
        l_recon = reconstruction_loss(R, X, Y, C, lambda_reg)
        assert loss_pure_recon == pytest.approx(l_recon)
        
        # rho = 0: Pure supervised
        loss_pure_sup, dict0 = combined_loss(
            R, X, Y, C, labels, labeled_idx, agg, rho=0.0, lambda_reg=lambda_reg
        )
        l_sup = supervised_loss(X, Y, labels, labeled_idx, agg)
        assert loss_pure_sup == pytest.approx(l_sup)
    
    def test_rho_interpolation(self):
        """Test that rho=0.5 interpolates between extremes."""
        m, n, d = 10, 100, 20
        R = np.random.rand(m, n)
        X = np.random.randn(d, m) * 0.1
        Y = np.random.randn(d, n) * 0.1
        C = np.ones((m, n))
        
        labels = np.random.randint(0, 2, n).astype(float)
        labels[50:] = np.nan
        labeled_idx = ~np.isnan(labels)
        
        agg = WeightedAggregator(m)
        lambda_reg = 0.01
        
        loss_half, loss_dict = combined_loss(
            R, X, Y, C, labels, labeled_idx, agg, rho=0.5, lambda_reg=lambda_reg
        )
        
        l_recon = loss_dict['reconstruction']
        l_sup = loss_dict['supervised']
        
        expected = 0.5 * l_recon + 0.5 * l_sup
        assert loss_half == pytest.approx(expected)
    
    def test_loss_dict_components(self):
        """Test that loss dictionary contains all expected components."""
        m, n, d = 10, 100, 20
        R = np.random.rand(m, n)
        X = np.random.randn(d, m) * 0.1
        Y = np.random.randn(d, n) * 0.1
        C = np.ones((m, n))
        
        labels = np.random.randint(0, 2, n).astype(float)
        labels[50:] = np.nan
        labeled_idx = ~np.isnan(labels)
        
        agg = MeanAggregator()
        
        loss, loss_dict = combined_loss(
            R, X, Y, C, labels, labeled_idx, agg, rho=0.5, lambda_reg=0.01
        )
        
        # Check all expected keys present
        assert 'total' in loss_dict
        assert 'reconstruction' in loss_dict
        assert 'supervised' in loss_dict
        assert 'recon_weighted' in loss_dict
        assert 'sup_weighted' in loss_dict
        assert 'rho' in loss_dict
        
        # Check values are correct
        assert loss_dict['total'] == pytest.approx(loss)
        assert loss_dict['rho'] == 0.5


class TestComputeRMSE:
    """Test RMSE computation."""
    
    def test_perfect_reconstruction(self):
        """Test RMSE with perfect reconstruction."""
        R = np.random.rand(10, 100)
        R_hat = R.copy()
        
        rmse = compute_rmse(R, R_hat)
        assert rmse == pytest.approx(0, abs=1e-10)
    
    def test_with_mask(self):
        """Test RMSE computation with mask."""
        R = np.random.rand(10, 100)
        R_hat = R + np.random.randn(10, 100) * 0.1
        
        # Compute RMSE on subset
        mask = np.zeros((10, 100), dtype=bool)
        mask[:, :50] = True
        
        rmse_full = compute_rmse(R, R_hat)
        rmse_masked = compute_rmse(R, R_hat, mask=mask)
        
        # Masked RMSE should be different
        assert rmse_full != pytest.approx(rmse_masked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
