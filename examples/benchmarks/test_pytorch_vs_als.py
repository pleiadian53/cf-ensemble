"""
Comprehensive Test: PyTorch vs ALS Trainer

Tests whether PyTorch's unified optimization avoids the aggregator weight collapse
that affects the ALS trainer with imbalanced data.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import generate_imbalanced_ensemble_data, EnsembleData
from cfensemble.optimization import CFEnsembleTrainer
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def test_pytorch_vs_als():
    """Compare PyTorch and ALS trainers on imbalanced data."""
    
    print("="*80)
    print("PYTORCH vs ALS: Comprehensive Test")
    print("="*80)
    
    # Generate high-quality synthetic data
    print("\n[1] Generating data...")
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        n_instances=500,
        n_classifiers=10,
        n_labeled=250,
        positive_rate=0.10,
        target_quality=0.70,
        random_state=42
    )
    
    n = R.shape[1]
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    # Test/train split
    y_test = y_true[~labeled_mask]
    R_test = R[:, ~labeled_mask]
    
    print(f"  Total instances: {n}")
    print(f"  Labeled: {np.sum(labeled_mask)}")
    print(f"  Test: {np.sum(~labeled_mask)}")
    print(f"  Positive rate: {np.mean(y_true):.1%}")
    
    # Prepare data for trainers
    labels_all = labels.copy()
    labels_all[~labeled_mask] = np.nan
    data = EnsembleData(R, labels_all)
    
    # Baseline: Simple average
    print("\n[2] Baseline: Simple Average")
    simple_avg = np.mean(R_test, axis=0)
    pr_simple = average_precision_score(y_test, simple_avg)
    roc_simple = roc_auc_score(y_test, simple_avg)
    print(f"  PR-AUC:  {pr_simple:.4f}")
    print(f"  ROC-AUC: {roc_simple:.4f}")
    
    # Test 1: ALS without freeze (should fail)
    print("\n[3] ALS Trainer (no freeze) - CONTROL")
    trainer_als_no_freeze = CFEnsembleTrainer(
        n_classifiers=10,
        latent_dim=15,
        rho=0.5,
        lambda_reg=0.01,
        max_iter=100,
        freeze_aggregator_iters=0,  # NO FREEZE
        use_label_aware_confidence=True,
        verbose=False,
        random_seed=42
    )
    trainer_als_no_freeze.fit(data)
    preds_als_no_freeze = trainer_als_no_freeze.predict()[~labeled_mask]
    pr_als_no_freeze = average_precision_score(y_test, preds_als_no_freeze)
    roc_als_no_freeze = roc_auc_score(y_test, preds_als_no_freeze)
    
    w_no_freeze = trainer_als_no_freeze.aggregator.get_weights()
    print(f"  PR-AUC:  {pr_als_no_freeze:.4f}")
    print(f"  ROC-AUC: {roc_als_no_freeze:.4f}")
    print(f"  Weights: [{w_no_freeze[0]:.3f}, {w_no_freeze[1]:.3f}, ..., {w_no_freeze[-1]:.3f}]")
    print(f"  Weight std: {np.std(w_no_freeze):.4f}")
    print(f"  Prediction std: {np.std(preds_als_no_freeze):.4f}")
    
    if np.std(w_no_freeze) < 0.01:
        print(f"  Status: ❌ COLLAPSED (as expected)")
    else:
        print(f"  Status: ⚠️  Unexpected (weights didn't collapse)")
    
    # Test 2: ALS with freeze
    print("\n[4] ALS Trainer (freeze=50)")
    trainer_als_freeze = CFEnsembleTrainer(
        n_classifiers=10,
        latent_dim=15,
        rho=0.5,
        lambda_reg=0.01,
        max_iter=100,
        freeze_aggregator_iters=50,  # FREEZE
        use_label_aware_confidence=True,
        verbose=False,
        random_seed=42
    )
    trainer_als_freeze.fit(data)
    preds_als_freeze = trainer_als_freeze.predict()[~labeled_mask]
    pr_als_freeze = average_precision_score(y_test, preds_als_freeze)
    roc_als_freeze = roc_auc_score(y_test, preds_als_freeze)
    
    w_freeze = trainer_als_freeze.aggregator.get_weights()
    print(f"  PR-AUC:  {pr_als_freeze:.4f}")
    print(f"  ROC-AUC: {roc_als_freeze:.4f}")
    print(f"  Weights: [{w_freeze[0]:.3f}, {w_freeze[1]:.3f}, ..., {w_freeze[-1]:.3f}]")
    print(f"  Weight std: {np.std(w_freeze):.4f}")
    print(f"  Prediction std: {np.std(preds_als_freeze):.4f}")
    
    if pr_als_freeze > pr_simple * 0.9:
        print(f"  Status: ✅ Works (good PR-AUC)")
    else:
        print(f"  Status: ❌ Failed")
    
    # Test 3: PyTorch trainer
    print("\n[5] PyTorch Trainer (unified optimization)")
    
    try:
        trainer_pytorch = CFEnsemblePyTorchTrainer(
            n_classifiers=10,
            latent_dim=15,
            rho=0.5,
            lambda_reg=0.01,
            max_epochs=200,
            lr=0.01,
            optimizer='adam',
            patience=30,
            aggregator_type='weighted',
            verbose=False,
            random_seed=42
        )
        trainer_pytorch.fit(data)
        preds_pytorch = trainer_pytorch.predict()[~labeled_mask]
        pr_pytorch = average_precision_score(y_test, preds_pytorch)
        roc_pytorch = roc_auc_score(y_test, preds_pytorch)
        
        w_pytorch, b_pytorch = trainer_pytorch.get_aggregator_weights()
        print(f"  PR-AUC:  {pr_pytorch:.4f}")
        print(f"  ROC-AUC: {roc_pytorch:.4f}")
        print(f"  Weights: [{w_pytorch[0]:.3f}, {w_pytorch[1]:.3f}, ..., {w_pytorch[-1]:.3f}]")
        print(f"  Bias: {b_pytorch:.3f}")
        print(f"  Weight std: {np.std(w_pytorch):.4f}")
        print(f"  Prediction std: {np.std(preds_pytorch):.4f}")
        
        if pr_pytorch > pr_simple * 0.9 and np.std(w_pytorch) > 0.05:
            print(f"  Status: ✅ WORKING! (good PR-AUC + healthy weights)")
        elif pr_pytorch > pr_simple * 0.9:
            print(f"  Status: ⚠️  Good PR-AUC but small weights")
        else:
            print(f"  Status: ❌ Failed")
        
        pytorch_available = True
    
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        pytorch_available = False
        pr_pytorch = np.nan
        roc_pytorch = np.nan
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<25} {'PR-AUC':>10} {'ROC-AUC':>10} {'vs Baseline':>12} {'Status':>15}")
    print("-"*80)
    
    print(f"{'Simple Average':<25} {pr_simple:>10.4f} {roc_simple:>10.4f} {'--':>12} {'Baseline':>15}")
    
    vs_simple_no_freeze = (pr_als_no_freeze / pr_simple - 1) * 100
    status_no_freeze = "❌ Collapsed"
    print(f"{'ALS (no freeze)':<25} {pr_als_no_freeze:>10.4f} {roc_als_no_freeze:>10.4f} "
          f"{vs_simple_no_freeze:>11.1f}% {status_no_freeze:>15}")
    
    vs_simple_freeze = (pr_als_freeze / pr_simple - 1) * 100
    status_freeze = "✅ Works" if pr_als_freeze > pr_simple * 0.9 else "❌ Failed"
    print(f"{'ALS (freeze=50)':<25} {pr_als_freeze:>10.4f} {roc_als_freeze:>10.4f} "
          f"{vs_simple_freeze:>11.1f}% {status_freeze:>15}")
    
    if pytorch_available:
        vs_simple_pytorch = (pr_pytorch / pr_simple - 1) * 100
        if pr_pytorch > pr_simple * 0.9 and np.std(w_pytorch) > 0.05:
            status_pytorch = "✅ Best"
        elif pr_pytorch > pr_simple * 0.9:
            status_pytorch = "⚠️  Small weights"
        else:
            status_pytorch = "❌ Failed"
        print(f"{'PyTorch (unified)':<25} {pr_pytorch:>10.4f} {roc_pytorch:>10.4f} "
              f"{vs_simple_pytorch:>11.1f}% {status_pytorch:>15}")
    else:
        print(f"{'PyTorch (unified)':<25} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'Not available':>15}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print("\n1. ALS without freeze:")
    print(f"   - Weight collapse: {np.std(w_no_freeze) < 0.01}")
    print(f"   - Performance degraded: {pr_als_no_freeze < pr_simple * 0.5}")
    
    print("\n2. ALS with freeze=50:")
    print(f"   - Achieves good PR-AUC: {pr_als_freeze > pr_simple * 0.9}")
    print(f"   - But weights still small: {np.std(w_freeze) < 0.01}")
    print(f"   - Prediction variance low: {np.std(preds_als_freeze) < 0.01}")
    
    if pytorch_available:
        print("\n3. PyTorch unified optimization:")
        print(f"   - Achieves good PR-AUC: {pr_pytorch > pr_simple * 0.9}")
        print(f"   - Weights healthy: {np.std(w_pytorch) > 0.05}")
        print(f"   - Prediction variance: {np.std(preds_pytorch):.4f}")
        print(f"   - Avoids collapse: {np.std(w_pytorch) > np.std(w_no_freeze)}")
        
        if pr_pytorch > pr_simple * 0.9 and np.std(w_pytorch) > 0.05:
            print("\n✅ CONCLUSION: PyTorch avoids weight collapse!")
        elif pr_pytorch > pr_simple * 0.9:
            print("\n⚠️  CONCLUSION: PyTorch achieves good PR-AUC but weights still small")
        else:
            print("\n❌ CONCLUSION: PyTorch also struggles with this problem")
    else:
        print("\n3. PyTorch: Not tested (import failed)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    test_pytorch_vs_als()
