"""
Ultra-Detailed Diagnostic for CF-Ensemble

Tests EVERYTHING to find why both ALS and PyTorch are failing.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import EnsembleData
from cfensemble.optimization import CFEnsembleTrainer, PYTORCH_AVAILABLE
if PYTORCH_AVAILABLE:
    from cfensemble.optimization import CFEnsemblePyTorchTrainer


def test_trivial_perfect_data():
    """Test on the simplest possible data that MUST work."""
    print("="*70)
    print("TEST 1: Trivial Perfect Data")
    print("="*70)
    
    # 3 perfect classifiers, 20 instances
    # Classifier predictions ARE the true labels (no noise)
    np.random.seed(42)
    n = 20
    y_true = np.random.randint(0, 2, n).astype(float)
    
    # Perfect predictions with tiny noise
    R = np.array([y_true + np.random.randn(n) * 0.01,
                  y_true + np.random.randn(n) * 0.01,
                  y_true + np.random.randn(n) * 0.01])
    R = np.clip(R, 0, 1)
    
    # Half labeled
    labels = y_true.copy()
    labels[10:] = np.nan
    
    print(f"Data: {R.shape}")
    print(f"Base classifier mean error: {np.mean(np.abs(R - y_true)):.4f} (should be ~0.01)")
    print(f"Simple average PR-AUC vs truth: ", end="")
    from sklearn.metrics import average_precision_score
    pr_simple = average_precision_score(y_true[10:], np.mean(R[:, 10:], axis=0))
    print(f"{pr_simple:.3f}")
    
    # Create data
    data = EnsembleData(R, labels)
    
    # Test CF-ALS with mean aggregator (NO learnable weights)
    print("\n[A] CF-ALS, mean aggregator, ρ=1.0 (pure reconstruction)...")
    trainer_als = CFEnsembleTrainer(
        n_classifiers=3,
        latent_dim=2,
        rho=1.0,  # Pure reconstruction
        lambda_reg=0.001,  # Very small reg
        max_iter=50,
        aggregator_type='mean',  # NO learning
        verbose=False
    )
    trainer_als.fit(data)
    
    preds_als = trainer_als.predict()
    pr_als = average_precision_score(y_true[10:], preds_als[10:])
    
    # Check reconstruction
    R_hat = trainer_als.get_reconstruction()
    rmse = np.sqrt(np.mean((R - R_hat)**2))
    
    print(f"  Converged: {trainer_als.converged_}")
    print(f"  Reconstruction RMSE: {rmse:.6f} (should be <0.01 for perfect data)")
    print(f"  PR-AUC: {pr_als:.3f}")
    print(f"  X norms: {np.linalg.norm(trainer_als.X, axis=0)}")
    print(f"  Y norms (first 5): {np.linalg.norm(trainer_als.Y, axis=0)[:5]}")
    
    if pr_als < pr_simple - 0.1:
        print("  ❌ FAIL: Worse than simple average on trivial data!")
        print("     This suggests matrix factorization itself is broken")
    else:
        print("  ✅ PASS: Reconstruction works on trivial data")
    
    return pr_als, pr_simple


def test_aggregator_alone():
    """Test if aggregator can learn from fixed good factors."""
    print("\n" + "="*70)
    print("TEST 2: Aggregator Learning (Fixed Factors)")
    print("="*70)
    
    np.random.seed(42)
    m, d, n = 5, 3, 100
    
    # Create GOOD factors that encode the true labels
    X = np.random.randn(d, m) * 0.5
    Y = np.random.randn(d, n) * 0.5
    
    # True labels from first latent dimension
    y_true = (Y[0, :] > 0).astype(float)
    
    # Reconstruct probabilities (with noise)
    R_hat = 1 / (1 + np.exp(-(X.T @ Y)))  # Sigmoid
    
    print(f"Data: {m} classifiers, {n} instances")
    print(f"Simple average AUC: ", end="")
    from sklearn.metrics import roc_auc_score
    auc_simple = roc_auc_score(y_true, np.mean(R_hat, axis=0))
    print(f"{auc_simple:.3f}")
    
    # Test weighted aggregator
    from cfensemble.ensemble.aggregators import WeightedAggregator
    agg = WeightedAggregator(m, init_uniform=True)
    
    labeled_idx = np.ones(n, dtype=bool)
    
    print("\nTraining aggregator (20 iterations)...")
    for i in range(20):
        agg.update(X, Y, labeled_idx, y_true, lr=0.1)
        
        if i % 5 == 0:
            y_pred = agg.predict(R_hat)
            auc = roc_auc_score(y_true, y_pred)
            print(f"  Iter {i:2d}: AUC={auc:.3f}, w={agg.w.round(3)}, b={agg.b:.3f}")
    
    y_pred_final = agg.predict(R_hat)
    auc_final = roc_auc_score(y_true, y_pred_final)
    
    if auc_final > auc_simple + 0.05:
        print(f"\n✅ PASS: Aggregator learned (AUC {auc_simple:.3f} → {auc_final:.3f})")
    else:
        print(f"\n❌ FAIL: Aggregator didn't improve ({auc_simple:.3f} → {auc_final:.3f})")


def test_reconstruction_quality():
    """Test if matrix factorization works at all."""
    print("\n" + "="*70)
    print("TEST 3: Pure Matrix Factorization")
    print("="*70)
    
    # Known low-rank matrix
    np.random.seed(42)
    d_true = 2
    m, n = 5, 50
    
    X_true = np.random.randn(d_true, m) * 0.3
    Y_true = np.random.randn(d_true, n) * 0.3
    R = np.clip(X_true.T @ Y_true + 0.5, 0, 1)
    
    print(f"True rank: {d_true}")
    print(f"Matrix shape: {R.shape}")
    print(f"Value range: [{R.min():.2f}, {R.max():.2f}]")
    
    # Test different latent dims
    for latent_dim in [2, 3, 5]:
        labels = np.full(n, np.nan)  # All unlabeled
        data = EnsembleData(R, labels)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=latent_dim,
            rho=1.0,  # Pure reconstruction
            lambda_reg=0.01,
            max_iter=100,
            aggregator_type='mean',
            verbose=False
        )
        
        trainer.fit(data)
        
        R_hat = trainer.get_reconstruction()
        rmse = np.sqrt(np.mean((R - R_hat)**2))
        
        status = "✓" if rmse < 0.05 else "✗"
        print(f"  d={latent_dim}: RMSE={rmse:.6f}, Converged={trainer.converged_} {status}")
        
        if latent_dim == d_true and rmse > 0.05:
            print("    ❌ FAIL: Can't recover known low-rank matrix!")
            print("       ALS implementation may have bugs")


def test_end_to_end_minimal():
    """Minimal end-to-end test."""
    print("\n" + "="*70)
    print("TEST 4: Minimal End-to-End")
    print("="*70)
    
    # Tiny dataset
    np.random.seed(42)
    m, n = 3, 20
    
    # Good classifiers (0.8 accuracy)
    y_true = np.random.randint(0, 2, n).astype(float)
    R = np.zeros((m, n))
    for u in range(m):
        correct = np.random.rand(n) < 0.8
        R[u, :] = np.where(correct, y_true + np.random.randn(n)*0.1, 
                                     1-y_true + np.random.randn(n)*0.1)
    R = np.clip(R, 0, 1)
    
    # Half labeled
    labels = y_true.copy()
    labels[10:] = np.nan
    
    print(f"Data: {m} classifiers × {n} instances (10 train, 10 test)")
    print(f"Base classifier accuracies: ", end="")
    for u in range(m):
        acc = np.mean((R[u, :10] > 0.5) == y_true[:10])
        print(f"{acc:.2f}", end=" ")
    print()
    
    # Simple average baseline
    from sklearn.metrics import average_precision_score
    pr_simple = average_precision_score(y_true[10:], np.mean(R[:, 10:], axis=0))
    print(f"Simple average PR-AUC: {pr_simple:.3f}")
    
    # Test CF-ALS
    data = EnsembleData(R, labels)
    trainer = CFEnsembleTrainer(
        n_classifiers=m,
        latent_dim=2,
        rho=0.5,
        lambda_reg=0.01,
        max_iter=50,
        aggregator_type='mean',
        use_label_aware_confidence=True,
        verbose=True  # VERBOSE to see what happens
    )
    
    print("\nTraining CF-ALS...")
    trainer.fit(data)
    
    preds = trainer.predict()
    pr_cf = average_precision_score(y_true[10:], preds[10:])
    
    print(f"\nCF-ALS PR-AUC: {pr_cf:.3f}")
    
    if pr_cf < pr_simple - 0.1:
        print("❌ CF-ALS worse than simple average on trivial data")
        print("   Investigating...")
        
        # Check reconstruction
        R_hat = trainer.get_reconstruction()
        print(f"\n   R_hat range: [{R_hat.min():.3f}, {R_hat.max():.3f}]")
        print(f"   R_hat mean: {R_hat.mean():.3f}")
        print(f"   Reconstruction RMSE: {np.sqrt(np.mean((R - R_hat)**2)):.4f}")
        
        # Check if predictions make sense
        print(f"\n   Predictions range: [{preds.min():.3f}, {preds.max():.3f}]")
        print(f"   Predictions std: {np.std(preds):.3f}")
        
        # Check factors
        print(f"\n   X norms: {np.linalg.norm(trainer.X, axis=0)}")
        print(f"   Y norms (first 5): {np.linalg.norm(trainer.Y, axis=0)[:5]}")
        
    else:
        print("✅ CF-ALS works on trivial data!")


if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# ULTRA-DIAGNOSTIC: Find Why CF-Ensemble Fails")
    print("#"*70)
    
    # Run tests
    test_trivial_perfect_data()
    test_aggregator_alone()
    test_reconstruction_quality()
    test_end_to_end_minimal()
    
    print("\n" + "="*70)
    print("Diagnostic complete. Check results above.")
    print("="*70)
