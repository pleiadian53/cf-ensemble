"""
Diagnostic script to debug CF-Ensemble training issues.

Tests on simple, balanced data to isolate the problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import EnsembleData, generate_imbalanced_ensemble_data
from cfensemble.optimization import CFEnsembleTrainer


def test_on_simple_data():
    """Test CF-Ensemble on easy, balanced data."""
    print("="*60)
    print("DIAGNOSTIC: Testing CF-Ensemble on Simple Data")
    print("="*60)
    
    # Generate EASY, BALANCED data
    print("\n1. Generating simple synthetic data...")
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        n_instances=200,  # Small dataset
        n_classifiers=10,  # Fewer classifiers
        n_labeled=100,  # 50% labeled
        positive_rate=0.50,  # BALANCED (not imbalanced)
        target_quality=0.85,  # HIGH quality classifiers
        diversity='low',  # LOW diversity (easier)
        random_state=42
    )
    
    # Convert labeled_idx from indices to boolean mask
    n = R.shape[1]
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    n_train = labeled_mask.sum()
    n_test = (~labeled_mask).sum()
    
    print(f"  Data shape: {R.shape}")
    print(f"  Train: {n_train}, Test: {n_test}")
    print(f"  Positive rate: {labels[labeled_mask].mean():.2f}")
    # Check classifier accuracies
    accs = []
    for i in range(min(3, R.shape[0])):
        acc = ((R[i, labeled_mask] >= 0.5) == labels[labeled_mask]).mean()
        accs.append(f"{acc:.2f}")
    print(f"  Base classifier quality (first 3): {', '.join(accs)}")
    
    # Split train/test
    R_train = R[:, labeled_mask]
    R_test = R[:, ~labeled_mask]
    y_train = labels[labeled_mask]
    y_test = y_true[~labeled_mask]
    
    # Combine for transductive learning
    R_combined = np.hstack([R_train, R_test])
    labels_combined = np.concatenate([y_train, np.full(n_test, np.nan)])
    
    print(f"\n2. Testing different hyperparameters...")
    print(f"{'='*60}")
    
    configs = [
        {'name': 'Default', 'latent_dim': 20, 'lambda_reg': 0.01, 'rho': 0.5},
        {'name': 'Less Reg', 'latent_dim': 20, 'lambda_reg': 0.001, 'rho': 0.5},
        {'name': 'Smaller Dim', 'latent_dim': 5, 'lambda_reg': 0.01, 'rho': 0.5},
        {'name': 'Pure Recon', 'latent_dim': 10, 'lambda_reg': 0.01, 'rho': 1.0},
        {'name': 'Pure Sup', 'latent_dim': 10, 'lambda_reg': 0.01, 'rho': 0.0},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  latent_dim={config['latent_dim']}, lambda_reg={config['lambda_reg']}, rho={config['rho']}")
        
        ensemble_data = EnsembleData(R_combined, labels_combined)
        
        trainer = CFEnsembleTrainer(
            n_classifiers=R.shape[0],
            latent_dim=config['latent_dim'],
            rho=config['rho'],
            lambda_reg=config['lambda_reg'],
            max_iter=100,
            tol=1e-4,
            aggregator_lr=0.1,
            verbose=False
        )
        
        trainer.fit(ensemble_data)
        
        # Get predictions
        all_preds = trainer.predict()
        y_pred = all_preds[n_train:]
        
        # Evaluate
        from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
        pr_auc = average_precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred >= 0.5)
        
        # Get losses
        recon_loss = trainer.history['reconstruction'][-1] if trainer.history['reconstruction'] else np.nan
        sup_loss = trainer.history['supervised'][-1] if trainer.history['supervised'] else np.nan
        
        # Reconstruction error
        R_hat = trainer.get_reconstruction()
        rmse = np.sqrt(np.mean((R_combined - R_hat)**2))
        
        print(f"  Converged: {'✓' if trainer.converged_ else '✗'} ({trainer.n_iter_} iter)")
        print(f"  PR-AUC: {pr_auc:.3f}, ROC-AUC: {roc_auc:.3f}, Acc: {acc:.3f}")
        print(f"  Recon Loss: {recon_loss:.4f}, Sup Loss: {sup_loss:.4f}")
        print(f"  Reconstruction RMSE: {rmse:.4f}")
        
        results.append({
            **config,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'acc': acc,
            'recon_loss': recon_loss,
            'sup_loss': sup_loss,
            'rmse': rmse,
            'converged': trainer.converged_,
            'n_iter': trainer.n_iter_
        })
    
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON:")
    print(f"{'='*60}")
    
    # Simple average
    y_pred_simple = np.mean(R_test, axis=0)
    pr_auc_simple = average_precision_score(y_test, y_pred_simple)
    roc_auc_simple = roc_auc_score(y_test, y_pred_simple)
    acc_simple = accuracy_score(y_test, y_pred_simple >= 0.5)
    print(f"Simple Average: PR-AUC={pr_auc_simple:.3f}, ROC-AUC={roc_auc_simple:.3f}, Acc={acc_simple:.3f}")
    
    # Stacking
    from sklearn.linear_model import LogisticRegression
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(R_train.T, y_train)
    y_pred_stack = stacker.predict_proba(R_test.T)[:, 1]
    pr_auc_stack = average_precision_score(y_test, y_pred_stack)
    roc_auc_stack = roc_auc_score(y_test, y_pred_stack)
    acc_stack = accuracy_score(y_test, y_pred_stack >= 0.5)
    print(f"Stacking: PR-AUC={pr_auc_stack:.3f}, ROC-AUC={roc_auc_stack:.3f}, Acc={acc_stack:.3f}")
    
    print(f"\n{'='*60}")
    print("DIAGNOSIS:")
    print(f"{'='*60}")
    
    best_cf = max(results, key=lambda x: x['pr_auc'])
    print(f"Best CF config: {best_cf['name']}")
    print(f"  PR-AUC: {best_cf['pr_auc']:.3f}")
    
    if best_cf['pr_auc'] < pr_auc_simple:
        print("\n❌ CF-Ensemble WORSE than simple average!")
        print("   Possible issues:")
        print("   1. Reconstruction quality is poor (high RMSE)")
        print("   2. Aggregator not learning properly")
        print("   3. Hyperparameters need more tuning")
        print(f"   4. Current best RMSE: {best_cf['rmse']:.4f} (should be <0.1)")
    elif best_cf['pr_auc'] < pr_auc_stack:
        print("\n⚠️  CF-Ensemble better than simple average but worse than stacking")
        print("   This is expected on easy data")
        print("   CF-Ensemble should shine on harder, more imbalanced data")
    else:
        print("\n✓ CF-Ensemble working! Better than both baselines")
    
    # Check convergence
    non_converged = [r for r in results if not r['converged']]
    if len(non_converged) == len(results):
        print("\n❌ NONE of the configs converged!")
        print("   This suggests a fundamental optimization issue")
    elif non_converged:
        print(f"\n⚠️  {len(non_converged)}/{len(results)} configs didn't converge")
    
    return results


def test_reconstruction_only():
    """Test pure reconstruction (ρ=1.0) to isolate matrix factorization."""
    print("\n" + "="*60)
    print("DIAGNOSTIC: Testing Pure Reconstruction (ρ=1.0)")
    print("="*60)
    
    # Simple known matrix
    print("\n1. Testing on a simple low-rank matrix...")
    
    # Create a known low-rank matrix
    np.random.seed(42)
    d = 3
    m, n = 10, 100
    X_true = np.random.randn(d, m) * 0.3
    Y_true = np.random.randn(d, n) * 0.3
    R = np.clip(X_true.T @ Y_true + 0.5, 0, 1)  # (m, n) in [0, 1]
    
    print(f"  Matrix shape: {R.shape}")
    print(f"  True rank: {d}")
    print(f"  Value range: [{R.min():.2f}, {R.max():.2f}]")
    
    # Fit CF-Ensemble with pure reconstruction
    labels = np.full(n, np.nan)  # All unlabeled
    ensemble_data = EnsembleData(R, labels)
    
    for latent_dim in [3, 5, 10]:
        print(f"\n  Testing latent_dim={latent_dim}...")
        
        trainer = CFEnsembleTrainer(
            n_classifiers=m,
            latent_dim=latent_dim,
            rho=1.0,  # Pure reconstruction
            lambda_reg=0.01,
            max_iter=100,
            verbose=False
        )
        
        trainer.fit(ensemble_data)
        
        R_hat = trainer.get_reconstruction()
        rmse = np.sqrt(np.mean((R - R_hat)**2))
        
        print(f"    Converged: {'✓' if trainer.converged_ else '✗'} ({trainer.n_iter_} iter)")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    Recon Loss: {trainer.history['reconstruction'][-1]:.6f}")
        
        if rmse < 0.01 and latent_dim >= d:
            print("    ✓ Good reconstruction (RMSE < 0.01)")
        elif rmse < 0.1:
            print("    ⚠️  Acceptable reconstruction (0.01 < RMSE < 0.1)")
        else:
            print("    ❌ Poor reconstruction (RMSE > 0.1) - ALS may have bugs")


if __name__ == '__main__':
    # Test 1: Simple balanced data
    results = test_on_simple_data()
    
    # Test 2: Pure reconstruction on known matrix
    test_reconstruction_only()
    
    print("\n" + "="*60)
    print("Diagnostic complete!")
    print("="*60)
