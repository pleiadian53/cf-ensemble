"""
Tests for PyTorch gradient descent optimization and comparison with ALS.
"""

import pytest
import numpy as np
import torch

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

from cfensemble.optimization.pytorch_gd import PyTorchCFOptimizer, compare_als_vs_pytorch
from cfensemble.optimization.als import (
    update_classifier_factors,
    update_instance_factors,
    compute_reconstruction_error
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    m, n = 5, 20
    d = 3
    
    # Generate true factors
    X_true = np.random.randn(d, m) * 0.5
    Y_true = np.random.randn(d, n) * 0.5
    
    # Generate probability matrix with noise
    R = X_true.T @ Y_true
    R = np.clip(R + np.random.randn(m, n) * 0.1, 0.01, 0.99)
    
    # Confidence weights
    C = np.abs(R - 0.5)
    
    return R, C, m, n, d


class TestPyTorchCFOptimizer:
    """Test PyTorch optimizer initialization and basic functionality."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = PyTorchCFOptimizer(m=5, n=20, latent_dim=3)
        
        assert optimizer.X.shape == (3, 5)
        assert optimizer.Y.shape == (3, 20)
        assert optimizer.device == 'cpu'
        assert not optimizer._is_fitted
    
    def test_initialization_with_seed(self):
        """Test reproducible initialization with seed."""
        opt1 = PyTorchCFOptimizer(m=5, n=20, latent_dim=3, random_seed=42)
        opt2 = PyTorchCFOptimizer(m=5, n=20, latent_dim=3, random_seed=42)
        
        X1, Y1 = opt1.get_factors()
        X2, Y2 = opt2.get_factors()
        
        assert np.allclose(X1, X2)
        assert np.allclose(Y1, Y2)
    
    def test_different_optimizers(self, sample_data):
        """Test different optimizer types."""
        R, C, m, n, d = sample_data
        
        for opt_type in ['adam', 'sgd', 'adamw']:
            optimizer = PyTorchCFOptimizer(
                m=m, n=n, latent_dim=d,
                optimizer_type=opt_type,
                random_seed=42
            )
            optimizer.fit(R, C, max_iter=10, verbose=False)
            
            assert optimizer._is_fitted
            assert len(optimizer.history['loss']) == 10


class TestPyTorchFitting:
    """Test PyTorch optimizer fitting."""
    
    def test_basic_fitting(self, sample_data):
        """Test basic fitting decreases loss."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, random_seed=42)
        optimizer.fit(R, C, max_iter=20, verbose=False)
        
        # Loss should decrease
        assert optimizer.history['loss'][0] > optimizer.history['loss'][-1]
        
        # Should be fitted
        assert optimizer._is_fitted
    
    def test_convergence(self, sample_data):
        """Test convergence criterion."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(
            m=m, n=n, latent_dim=d,
            random_seed=42,
            lr=0.1  # Higher LR for faster convergence
        )
        optimizer.fit(R, C, max_iter=100, tol=1e-5, verbose=False)
        
        # Should converge before max_iter
        assert len(optimizer.history['loss']) < 100
    
    def test_reconstruction_quality(self, sample_data):
        """Test that reconstruction improves."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, random_seed=42)
        
        # Initial reconstruction error
        initial_error = optimizer.reconstruction_error(R, C)
        
        # Fit
        optimizer.fit(R, C, max_iter=50, verbose=False)
        
        # Final reconstruction error
        final_error = optimizer.reconstruction_error(R, C)
        
        # Should improve significantly
        assert final_error < initial_error * 0.5


class TestALSvsPyTorch:
    """Test consistency between ALS and PyTorch."""
    
    def test_small_problem_consistency(self):
        """Test that ALS and PyTorch converge to similar solutions."""
        np.random.seed(42)
        m, n, d = 5, 20, 3
        
        # Generate data
        R = np.random.rand(m, n)
        C = np.abs(R - 0.5)
        
        lambda_reg = 0.01
        max_iter = 50
        
        # ALS
        np.random.seed(42)
        X_als = np.random.randn(d, m) * 0.01
        Y_als = np.random.randn(d, n) * 0.01
        
        for _ in range(max_iter):
            X_als = update_classifier_factors(Y_als, R, C, lambda_reg)
            Y_als = update_instance_factors(X_als, R, C, lambda_reg)
        
        recon_error_als = compute_reconstruction_error(X_als, Y_als, R, C)
        
        # PyTorch
        optimizer = PyTorchCFOptimizer(
            m=m, n=n, latent_dim=d,
            lambda_reg=lambda_reg,
            lr=0.01,
            random_seed=42
        )
        optimizer.fit(R, C, max_iter=max_iter, verbose=False)
        
        X_pytorch, Y_pytorch = optimizer.get_factors()
        recon_error_pytorch = optimizer.reconstruction_error(R, C)
        
        # Should have similar reconstruction errors (within 10%)
        relative_diff = abs(recon_error_als - recon_error_pytorch) / recon_error_als
        assert relative_diff < 0.1, f"Reconstruction errors differ by {relative_diff*100:.1f}%"
    
    def test_compare_als_vs_pytorch_function(self):
        """Test the comparison utility function."""
        np.random.seed(42)
        R = np.random.rand(5, 20)
        C = np.abs(R - 0.5)
        
        results = compare_als_vs_pytorch(
            R, C,
            latent_dim=3,
            max_iter=30,
            random_seed=42,
            verbose=False
        )
        
        # Check all expected keys
        assert 'X_als' in results
        assert 'Y_als' in results
        assert 'X_pytorch' in results
        assert 'Y_pytorch' in results
        assert 'recon_error_als' in results
        assert 'recon_error_pytorch' in results
        assert 'final_loss_als' in results
        assert 'final_loss_pytorch' in results
        assert 'X_correlation' in results
        assert 'Y_correlation' in results
        
        # Reconstruction errors should be close
        relative_diff = abs(
            results['recon_error_als'] - results['recon_error_pytorch']
        ) / results['recon_error_als']
        assert relative_diff < 0.15  # Within 15%
    
    def test_convergence_comparison(self):
        """Compare convergence behavior."""
        np.random.seed(42)
        R = np.random.rand(5, 20)
        C = np.abs(R - 0.5)
        
        results = compare_als_vs_pytorch(
            R, C,
            latent_dim=3,
            max_iter=50,
            random_seed=42,
            verbose=False
        )
        
        # Both should decrease loss
        assert results['als_losses'][0] > results['als_losses'][-1]
        assert results['pytorch_losses'][0] > results['pytorch_losses'][-1]
        
        # Final losses should be similar
        relative_diff = abs(
            results['final_loss_als'] - results['final_loss_pytorch']
        ) / results['final_loss_als']
        assert relative_diff < 0.15  # Within 15%


class TestPyTorchPrediction:
    """Test prediction functionality."""
    
    def test_predict_shape(self, sample_data):
        """Test prediction output shape."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, random_seed=42)
        optimizer.fit(R, C, max_iter=10, verbose=False)
        
        R_hat = optimizer.predict()
        
        assert R_hat.shape == (m, n)
    
    def test_predict_range(self, sample_data):
        """Test that predictions are in reasonable range."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, random_seed=42)
        optimizer.fit(R, C, max_iter=50, verbose=False)
        
        R_hat = optimizer.predict()
        
        # Should be similar to original R (in [0, 1] range)
        # Note: Predictions might go slightly outside [0, 1] since we don't constrain
        assert np.all(R_hat > -0.5)
        assert np.all(R_hat < 1.5)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_classifier(self):
        """Test with single classifier."""
        np.random.seed(42)
        R = np.random.rand(1, 10)
        C = np.abs(R - 0.5)
        
        optimizer = PyTorchCFOptimizer(m=1, n=10, latent_dim=2, random_seed=42)
        optimizer.fit(R, C, max_iter=20, verbose=False)
        
        assert optimizer._is_fitted
        X, Y = optimizer.get_factors()
        assert X.shape == (2, 1)
        assert Y.shape == (2, 10)
    
    def test_uniform_confidence(self):
        """Test with uniform confidence weights."""
        np.random.seed(42)
        R = np.random.rand(5, 20)
        C = np.ones_like(R)  # Uniform confidence
        
        optimizer = PyTorchCFOptimizer(m=5, n=20, latent_dim=3, random_seed=42)
        optimizer.fit(R, C, max_iter=30, verbose=False)
        
        assert optimizer._is_fitted
    
    def test_high_regularization(self):
        """Test with high regularization (should shrink factors)."""
        np.random.seed(42)
        R = np.random.rand(5, 20)
        C = np.abs(R - 0.5)
        
        # Low regularization
        opt_low = PyTorchCFOptimizer(m=5, n=20, latent_dim=3, lambda_reg=0.001, random_seed=42)
        opt_low.fit(R, C, max_iter=50, verbose=False)
        X_low, Y_low = opt_low.get_factors()
        
        # High regularization
        opt_high = PyTorchCFOptimizer(m=5, n=20, latent_dim=3, lambda_reg=1.0, random_seed=42)
        opt_high.fit(R, C, max_iter=50, verbose=False)
        X_high, Y_high = opt_high.get_factors()
        
        # High regularization should produce smaller norms
        assert np.linalg.norm(X_high) < np.linalg.norm(X_low)
        assert np.linalg.norm(Y_high) < np.linalg.norm(Y_low)


class TestDifferentDevices:
    """Test different device types (if available)."""
    
    def test_cpu_device(self, sample_data):
        """Test CPU device."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, device='cpu', random_seed=42)
        optimizer.fit(R, C, max_iter=10, verbose=False)
        
        assert optimizer._is_fitted
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, sample_data):
        """Test CUDA device if available."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, device='cuda', random_seed=42)
        optimizer.fit(R, C, max_iter=10, verbose=False)
        
        assert optimizer._is_fitted
        assert optimizer.X.device.type == 'cuda'
        assert optimizer.Y.device.type == 'cuda'
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device(self, sample_data):
        """Test MPS device (Apple Silicon) if available."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, device='mps', random_seed=42)
        optimizer.fit(R, C, max_iter=10, verbose=False)
        
        assert optimizer._is_fitted
        assert optimizer.X.device.type == 'mps'


class TestHistoryTracking:
    """Test training history tracking."""
    
    def test_history_length(self, sample_data):
        """Test history length matches iterations."""
        R, C, m, n, d = sample_data
        
        max_iter = 25
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, random_seed=42)
        optimizer.fit(R, C, max_iter=max_iter, verbose=False)
        
        # History length should match iterations
        assert len(optimizer.history['loss']) <= max_iter
        assert len(optimizer.history['reconstruction_loss']) <= max_iter
        assert len(optimizer.history['regularization_loss']) <= max_iter
    
    def test_history_decreasing(self, sample_data):
        """Test that loss generally decreases."""
        R, C, m, n, d = sample_data
        
        optimizer = PyTorchCFOptimizer(m=m, n=n, latent_dim=d, lr=0.05, random_seed=42)
        optimizer.fit(R, C, max_iter=50, verbose=False)
        
        # Loss should decrease overall (allow some fluctuation)
        initial_loss = optimizer.history['loss'][0]
        final_loss = optimizer.history['loss'][-1]
        
        assert final_loss < initial_loss * 0.8  # At least 20% improvement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
