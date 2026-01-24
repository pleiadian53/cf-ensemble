"""
Tests for CFEnsembleTrainer.
"""

import numpy as np
import pytest
import warnings
from src.cfensemble.data import EnsembleData
from src.cfensemble.optimization import CFEnsembleTrainer
from src.cfensemble.ensemble import WeightedAggregator


class TestTrainerInitialization:
    """Test CFEnsembleTrainer initialization."""
    
    def test_basic_initialization(self):
        """Test basic trainer initialization."""
        trainer = CFEnsembleTrainer(
            n_classifiers=10,
            latent_dim=20,
            rho=0.5,
            lambda_reg=0.01
        )
        
        assert trainer.n_classifiers == 10
        assert trainer.latent_dim == 20
        assert trainer.rho == 0.5
        assert trainer.lambda_reg == 0.01
        assert trainer.X is None  # Not fitted yet
        assert trainer.Y is None
    
    def test_invalid_rho(self):
        """Test that invalid rho values raise errors."""
        with pytest.raises(ValueError, match="rho must be in"):
            CFEnsembleTrainer(n_classifiers=10, rho=-0.1)
        
        with pytest.raises(ValueError, match="rho must be in"):
            CFEnsembleTrainer(n_classifiers=10, rho=1.5)
    
    def test_invalid_lambda(self):
        """Test that negative lambda raises error."""
        with pytest.raises(ValueError, match="lambda_reg must be non-negative"):
            CFEnsembleTrainer(n_classifiers=10, lambda_reg=-0.1)
    
    def test_invalid_latent_dim(self):
        """Test that invalid latent_dim raises error."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            CFEnsembleTrainer(n_classifiers=10, latent_dim=0)
    
    def test_aggregator_types(self):
        """Test different aggregator types."""
        # Mean aggregator
        trainer_mean = CFEnsembleTrainer(
            n_classifiers=10,
            aggregator_type='mean'
        )
        assert trainer_mean.aggregator is not None
        
        # Weighted aggregator
        trainer_weighted = CFEnsembleTrainer(
            n_classifiers=10,
            aggregator_type='weighted'
        )
        assert isinstance(trainer_weighted.aggregator, WeightedAggregator)
    
    def test_invalid_aggregator_type(self):
        """Test that invalid aggregator type raises error."""
        with pytest.raises(ValueError, match="Unknown aggregator type"):
            CFEnsembleTrainer(n_classifiers=10, aggregator_type='invalid')
    
    def test_random_seed(self):
        """Test reproducibility with random seed."""
        # Create dummy data ONCE (with fixed seed)
        np.random.seed(123)
        R = np.random.rand(10, 100)
        labels = np.random.randint(0, 2, 100).astype(float)
        data = EnsembleData(R, labels)
        
        # Create two trainers with same seed
        trainer1 = CFEnsembleTrainer(n_classifiers=10, random_seed=42, verbose=False)
        trainer2 = CFEnsembleTrainer(n_classifiers=10, random_seed=42, verbose=False)
        
        # Fit both
        trainer1.fit(data)
        trainer2.fit(data)
        
        # Should produce same results
        np.testing.assert_array_almost_equal(trainer1.X, trainer2.X)
        np.testing.assert_array_almost_equal(trainer1.Y, trainer2.Y)


class TestTrainerFitting:
    """Test CFEnsembleTrainer fitting."""
    
    def test_basic_fitting(self):
        """Test basic model fitting."""
        np.random.seed(42)
        
        # Create synthetic data
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        # Train
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer.fit(data)
        
        # Check that factors are initialized
        assert trainer.X is not None
        assert trainer.Y is not None
        assert trainer.X.shape == (5, m)
        assert trainer.Y.shape == (5, n)
        
        # Check history exists
        assert len(trainer.history['loss']) > 0
        assert len(trainer.history['reconstruction']) > 0
        assert len(trainer.history['supervised']) > 0
    
    def test_fitting_with_unlabeled_data(self):
        """Test fitting with partially labeled data."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        
        # Make 50% unlabeled
        labels[50:] = np.nan
        
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer.fit(data)
        
        assert trainer.X is not None
        assert len(trainer.history['loss']) > 0
    
    def test_fitting_without_labels(self):
        """Test fitting with no labels (pure reconstruction)."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.full(n, np.nan)  # All unlabeled
        
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            rho=0.5,  # Will be overridden to 1.0
            max_iter=20,
            verbose=False
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(data)
        
        # Should fall back to pure reconstruction
        assert trainer.rho == 1.0
        assert trainer.X is not None
    
    def test_convergence(self):
        """Test that loss decreases during training."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=30,
            verbose=False
        )
        trainer.fit(data)
        
        # Loss should decrease
        initial_loss = trainer.history['loss'][0]
        final_loss = trainer.history['loss'][-1]
        
        assert final_loss < initial_loss, \
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    
    def test_early_convergence(self):
        """Test early stopping when convergence criterion is met."""
        np.random.seed(42)
        
        m, n = 10, 50
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=100,
            tol=1e-3,  # Loose tolerance
            verbose=False
        )
        trainer.fit(data)
        
        # Should converge before max_iter
        assert trainer.converged_
        assert trainer.n_iter_ < 100
    
    def test_dimension_mismatch(self):
        """Test error when data dimensions don't match."""
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        # Create trainer with wrong number of classifiers
        trainer = CFEnsembleTrainer(
            n_classifiers=15,  # Mismatch!
            latent_dim=5,
            verbose=False
        )
        
        with pytest.raises(ValueError, match="Expected .* classifiers"):
            trainer.fit(data)
    
    def test_different_rho_values(self):
        """Test training with different rho values."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        # Pure reconstruction (rho=1.0)
        trainer_recon = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            rho=1.0,
            max_iter=20,
            verbose=False
        )
        trainer_recon.fit(data)
        
        # Balanced (rho=0.5)
        trainer_balanced = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            rho=0.5,
            max_iter=20,
            verbose=False
        )
        trainer_balanced.fit(data)
        
        # Pure supervised (rho=0.0)
        trainer_sup = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            rho=0.0,
            max_iter=20,
            verbose=False
        )
        trainer_sup.fit(data)
        
        # All should complete successfully
        assert trainer_recon.X is not None
        assert trainer_balanced.X is not None
        assert trainer_sup.X is not None


class TestTrainerPrediction:
    """Test CFEnsembleTrainer prediction."""
    
    def test_predict_without_fit(self):
        """Test that predicting without fitting raises error."""
        trainer = CFEnsembleTrainer(n_classifiers=10)
        
        with pytest.raises(ValueError, match="Model not fitted"):
            trainer.predict()
    
    def test_transductive_prediction(self):
        """Test transductive prediction (using fitted Y)."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer.fit(data)
        
        # Predict on training data
        predictions = trainer.predict(data)
        
        assert predictions.shape == (n,)
        assert np.all((predictions >= 0) & (predictions <= 1)), \
            "Predictions should be probabilities in [0, 1]"
    
    def test_inductive_prediction(self):
        """Test inductive prediction (new data)."""
        np.random.seed(42)
        
        m, n_train, n_test = 10, 100, 50
        R_train = np.random.rand(m, n_train)
        labels_train = np.random.randint(0, 2, n_train).astype(float)
        data_train = EnsembleData(R_train, labels_train)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer.fit(data_train)
        
        # New test data
        R_test = np.random.rand(m, n_test)
        
        # Predict on test data
        predictions = trainer.predict(R_new=R_test)
        
        assert predictions.shape == (n_test,)
        assert np.all((predictions >= 0) & (predictions <= 1))
    
    def test_inductive_prediction_wrong_dimensions(self):
        """Test error when test data has wrong dimensions."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=10,
            verbose=False
        )
        trainer.fit(data)
        
        # Test data with wrong number of classifiers
        R_test_wrong = np.random.rand(15, 50)  # 15 instead of 10
        
        with pytest.raises(ValueError, match="classifiers"):
            trainer.predict(R_new=R_test_wrong)
    
    def test_get_reconstruction(self):
        """Test getting reconstructed probability matrix."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer.fit(data)
        
        R_hat = trainer.get_reconstruction()
        
        assert R_hat.shape == (m, n)
        
        # Reconstruction should be reasonably close to original
        rmse = np.sqrt(np.mean((R - R_hat)**2))
        assert rmse < 0.5, f"Reconstruction RMSE should be reasonable, got {rmse:.4f}"
    
    def test_get_reconstruction_without_fit(self):
        """Test that get_reconstruction without fitting raises error."""
        trainer = CFEnsembleTrainer(n_classifiers=10)
        
        with pytest.raises(ValueError, match="Model not fitted"):
            trainer.get_reconstruction()


class TestTrainerUtilities:
    """Test CFEnsembleTrainer utility methods."""
    
    def test_get_aggregator_weights(self):
        """Test getting aggregator weights."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        # Weighted aggregator
        trainer_weighted = CFEnsembleTrainer(
            n_classifiers=m,
            aggregator_type='weighted',
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer_weighted.fit(data)
        
        weights = trainer_weighted.get_aggregator_weights()
        
        assert weights is not None
        assert weights.shape == (m,)
        assert np.allclose(np.sum(weights), 1.0), "Weights should sum to 1"
        
        # Mean aggregator
        trainer_mean = CFEnsembleTrainer(
            n_classifiers=m,
            aggregator_type='mean',
            latent_dim=5,
            max_iter=20,
            verbose=False
        )
        trainer_mean.fit(data)
        
        weights_mean = trainer_mean.get_aggregator_weights()
        assert weights_mean is None, "Mean aggregator has no weights"
    
    def test_history_tracking(self):
        """Test that training history is properly tracked."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=5,
            max_iter=10,
            verbose=False
        )
        trainer.fit(data)
        
        # Check all history fields exist
        assert 'loss' in trainer.history
        assert 'reconstruction' in trainer.history
        assert 'supervised' in trainer.history
        assert 'recon_weighted' in trainer.history
        assert 'sup_weighted' in trainer.history
        
        # Check they have same length
        n_iters = len(trainer.history['loss'])
        assert len(trainer.history['reconstruction']) == n_iters
        assert len(trainer.history['supervised']) == n_iters
        
        # Check weighted contributions sum to total
        for i in range(n_iters):
            total = trainer.history['loss'][i]
            weighted_sum = (trainer.history['recon_weighted'][i] + 
                          trainer.history['sup_weighted'][i])
            assert np.isclose(total, weighted_sum), \
                "Weighted contributions should sum to total loss"


class TestTrainerEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self):
        """Test complete training and prediction pipeline."""
        np.random.seed(42)
        
        # Create synthetic problem
        m, n_train, n_test = 10, 200, 100
        
        # Training data
        R_train = np.random.rand(m, n_train)
        labels_train = np.random.randint(0, 2, n_train).astype(float)
        labels_train[100:] = np.nan  # 50% unlabeled
        train_data = EnsembleData(R_train, labels_train)
        
        # Test data
        R_test = np.random.rand(m, n_test)
        
        # Train model
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=10,
            rho=0.5,
            lambda_reg=0.01,
            max_iter=30,
            verbose=False,
            random_seed=42
        )
        trainer.fit(train_data)
        
        # Predict on train and test
        pred_train = trainer.predict(train_data)
        pred_test = trainer.predict(R_new=R_test)
        
        # Basic sanity checks
        assert pred_train.shape == (n_train,)
        assert pred_test.shape == (n_test,)
        assert np.all((pred_train >= 0) & (pred_train <= 1))
        assert np.all((pred_test >= 0) & (pred_test <= 1))
        
        # Training loss should decrease
        assert trainer.history['loss'][-1] < trainer.history['loss'][0]
        
        # Aggregator weights should be learned
        weights = trainer.get_aggregator_weights()
        if weights is not None:
            assert not np.allclose(weights, 1/m), \
                "Weights should differ from uniform"
    
    def test_comparison_different_rho(self):
        """Test that different rho values produce different training behavior."""
        np.random.seed(42)
        
        m, n = 10, 100
        R = np.random.rand(m, n)
        labels = np.random.randint(0, 2, n).astype(float)
        data = EnsembleData(R, labels)
        
        # Train with rho=1.0 (reconstruction only)
        trainer1 = CFEnsembleTrainer(
            n_classifiers=m,
            rho=1.0,
            max_iter=50,  # More iterations
            verbose=False,
            random_seed=42
        )
        trainer1.fit(data)
        
        # Train with rho=0.0 (supervised only) with different seed
        trainer2 = CFEnsembleTrainer(
            n_classifiers=m,
            rho=0.0,  # Pure supervised (no reconstruction)
            max_iter=50,
            verbose=False,
            random_seed=123  # Different seed to ensure different trajectory
        )
        trainer2.fit(data)
        
        # Check that weighted contributions differ
        # trainer1 should have all weight on reconstruction
        recon1_weighted = trainer1.history['recon_weighted'][-1]
        sup1_weighted = trainer1.history['sup_weighted'][-1]
        
        # trainer2 should have all weight on supervised
        recon2_weighted = trainer2.history['recon_weighted'][-1]
        sup2_weighted = trainer2.history['sup_weighted'][-1]
        
        # For rho=1.0: recon_weighted should be non-zero, sup_weighted should be zero
        assert recon1_weighted > 0, "rho=1.0 should have positive reconstruction weight"
        assert sup1_weighted == 0 or sup1_weighted < 1e-10, "rho=1.0 should have zero supervised weight"
        
        # For rho=0.0: recon_weighted should be zero, sup_weighted should be non-zero
        assert recon2_weighted == 0 or recon2_weighted < 1e-10, "rho=0.0 should have zero reconstruction weight"
        assert sup2_weighted > 0, "rho=0.0 should have positive supervised weight"
        
        # The training should have completed successfully for both
        assert len(trainer1.history['loss']) > 0
        assert len(trainer2.history['loss']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
