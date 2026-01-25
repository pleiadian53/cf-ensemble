"""
DEEP DIAGNOSTIC: Why is CF-Ensemble destroying good predictions?

Traces every step to find the bug.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import EnsembleData, generate_imbalanced_ensemble_data
from cfensemble.optimization import CFEnsembleTrainer
from sklearn.metrics import average_precision_score, accuracy_score


def deep_diagnostic():
    """Trace CF-Ensemble training step-by-step."""
    print("="*70)
    print("DEEP DIAGNOSTIC: CF-Ensemble Training Trace")
    print("="*70)
    
    # Generate GOOD data
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        n_instances=100,  # Smaller for easier debugging
        n_classifiers=5,
        n_labeled=50,
        positive_rate=0.10,
        target_quality=0.70,
        random_state=42
    )
    
    n = R.shape[1]
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    y_train = labels[labeled_mask]
    y_test = y_true[~labeled_mask]
    
    print(f"\nData: 5 classifiers × 100 instances")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Check R quality
    print("\n1. BASE PREDICTIONS (R matrix):")
    print(f"   R shape: {R.shape}")
    print(f"   R range: [{R.min():.3f}, {R.max():.3f}]")
    print(f"   R mean: {R.mean():.3f}")
    
    # Simple average quality
    simple_avg = np.mean(R, axis=0)
    pr_simple_train = average_precision_score(y_train, simple_avg[labeled_mask])
    pr_simple_test = average_precision_score(y_test, simple_avg[~labeled_mask])
    print(f"\n2. SIMPLE AVERAGE QUALITY:")
    print(f"   Train PR-AUC: {pr_simple_train:.3f}")
    print(f"   Test PR-AUC:  {pr_simple_test:.3f}")
    
    # Prepare data for CF-Ensemble (transductive)
    R_all = R
    labels_all = labels.copy()
    labels_all[~labeled_mask] = np.nan
    
    data = EnsembleData(R_all, labels_all)
    
    # Train CF-Ensemble with VERBOSE
    print("\n3. CF-ENSEMBLE TRAINING:")
    print("="*70)
    
    trainer = CFEnsembleTrainer(
        n_classifiers=5,
        latent_dim=3,  # Small latent dim
        rho=0.5,
        lambda_reg=0.01,
        max_iter=10,  # Just 10 iterations for diagnosis
        aggregator_type='weighted',
        use_label_aware_confidence=True,
        aggregator_lr=0.1,
        verbose=True  # VERBOSE
    )
    
    trainer.fit(data)
    
    # Check reconstruction
    print("\n4. RECONSTRUCTION QUALITY:")
    R_hat = trainer.get_reconstruction()
    print(f"   R_hat shape: {R_hat.shape}")
    print(f"   R_hat range: [{R_hat.min():.3f}, {R_hat.max():.3f}]")
    print(f"   R_hat mean: {R_hat.mean():.3f}")
    
    rmse = np.sqrt(np.mean((R - R_hat)**2))
    print(f"   Reconstruction RMSE: {rmse:.6f}")
    
    # Compare R vs R_hat on test set
    simple_hat = np.mean(R_hat, axis=0)
    pr_hat_test = average_precision_score(y_test, simple_hat[~labeled_mask])
    print(f"   Simple avg of R_hat test PR-AUC: {pr_hat_test:.3f}")
    
    # Check factors
    print("\n5. LATENT FACTORS:")
    X, Y = trainer.X, trainer.Y
    print(f"   X norms: {np.linalg.norm(X, axis=0)}")
    print(f"   Y norms (first 5): {np.linalg.norm(Y, axis=0)[:5]}")
    print(f"   X mean: {X.mean():.3f}, std: {X.std():.3f}")
    print(f"   Y mean: {Y.mean():.3f}, std: {Y.std():.3f}")
    
    # Check if factors are degenerate
    X_var = np.var(X, axis=1)
    Y_var = np.var(Y, axis=1)
    print(f"   X variance per dim: {X_var}")
    print(f"   Y variance per dim: {Y_var}")
    
    if np.any(X_var < 1e-6) or np.any(Y_var < 1e-6):
        print("   ⚠️  WARNING: Some factors have near-zero variance!")
    
    # Check aggregator
    print("\n6. AGGREGATOR:")
    agg = trainer.aggregator
    if hasattr(agg, 'w') and hasattr(agg, 'b'):
        print(f"   Weights (w): {agg.w}")
        print(f"   Bias (b): {agg.b:.3f}")
        print(f"   Weight sum: {np.sum(agg.w):.3f}")
        print(f"   Weight std: {np.std(agg.w):.3f}")
        
        if np.std(agg.w) < 1e-6:
            print("   ⚠️  WARNING: Aggregator weights collapsed to same value!")
    
    # Final predictions
    print("\n7. FINAL PREDICTIONS:")
    preds_all = trainer.predict()
    print(f"   Prediction range: [{preds_all.min():.3f}, {preds_all.max():.3f}]")
    print(f"   Prediction mean: {preds_all.mean():.3f}")
    print(f"   Prediction std: {preds_all.std():.3f}")
    
    if preds_all.std() < 1e-6:
        print("   ❌ CRITICAL: All predictions are the same!")
        print("      This explains the catastrophic failure.")
    
    preds_train = preds_all[labeled_mask]
    preds_test = preds_all[~labeled_mask]
    
    pr_train = average_precision_score(y_train, preds_train)
    pr_test = average_precision_score(y_test, preds_test)
    
    print(f"\n   Train PR-AUC: {pr_train:.3f}")
    print(f"   Test PR-AUC:  {pr_test:.3f}")
    
    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    
    if pr_test < pr_simple_test - 0.1:
        print("❌ CF-Ensemble degraded performance significantly")
        
        if preds_all.std() < 1e-6:
            print("\nROOT CAUSE: Predictions collapsed to a constant")
            print("Possible reasons:")
            print("  1. Aggregator weights collapsed")
            print("  2. Reconstructed R_hat has no variance")
            print("  3. Latent factors degenerate")
        elif rmse > 0.3:
            print("\nROOT CAUSE: Poor reconstruction")
            print("  Matrix factorization failed to capture patterns")
        elif pr_hat_test < pr_simple_test - 0.1:
            print("\nROOT CAUSE: Reconstruction itself is bad")
            print("  ALS is degrading the base predictions")
        else:
            print("\nROOT CAUSE: Aggregator is broken")
            print("  Reconstruction is OK, but aggregator ruins it")
    else:
        print("✅ CF-Ensemble works correctly!")


if __name__ == '__main__':
    deep_diagnostic()
