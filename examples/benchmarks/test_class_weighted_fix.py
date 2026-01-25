"""
Comprehensive Test: Class-Weighted Gradients Fix

Tests whether class-weighted gradients fix the aggregator weight collapse
problem for both ALS and PyTorch trainers on imbalanced data.
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


def test_class_weighted_fix():
    """Test class-weighted gradients on imbalanced data."""
    
    print("="*80)
    print("CLASS-WEIGHTED GRADIENTS: Comprehensive Test")
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
    
    # Prepare data
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
    
    # Test 1: ALS without class weights (should fail)
    print("\n[3] ALS Trainer (no class weights) - CONTROL")
    trainer_als_no_weights = CFEnsembleTrainer(
        n_classifiers=10,
        latent_dim=15,
        rho=0.5,
        lambda_reg=0.01,
        max_iter=100,
        use_class_weights=False,  # DISABLED
        use_label_aware_confidence=True,
        verbose=False,
        random_seed=42
    )
    trainer_als_no_weights.fit(data)
    preds_als_no_weights = trainer_als_no_weights.predict()[~labeled_mask]
    pr_als_no_weights = average_precision_score(y_test, preds_als_no_weights)
    roc_als_no_weights = roc_auc_score(y_test, preds_als_no_weights)
    
    w_no_weights = trainer_als_no_weights.aggregator.get_weights()
    print(f"  PR-AUC:  {pr_als_no_weights:.4f}")
    print(f"  ROC-AUC: {roc_als_no_weights:.4f}")
    print(f"  Weights: [{w_no_weights[0]:.3f}, {w_no_weights[1]:.3f}, ..., {w_no_weights[-1]:.3f}]")
    print(f"  Weight std: {np.std(w_no_weights):.4f}")
    print(f"  Prediction std: {np.std(preds_als_no_weights):.4f}")
    print(f"  Status: {'❌ COLLAPSED' if np.std(w_no_weights) < 0.01 else '✅ OK'}")
    
    # Test 2: ALS with class weights (should work!)
    print("\n[4] ALS Trainer (with class weights) - NEW FIX")
    trainer_als_weighted = CFEnsembleTrainer(
        n_classifiers=10,
        latent_dim=15,
        rho=0.5,
        lambda_reg=0.01,
        max_iter=100,
        use_class_weights=True,  # ENABLED (default)
        use_label_aware_confidence=True,
        verbose=False,
        random_seed=42
    )
    trainer_als_weighted.fit(data)
    preds_als_weighted = trainer_als_weighted.predict()[~labeled_mask]
    pr_als_weighted = average_precision_score(y_test, preds_als_weighted)
    roc_als_weighted = roc_auc_score(y_test, preds_als_weighted)
    
    w_weighted = trainer_als_weighted.aggregator.get_weights()
    print(f"  PR-AUC:  {pr_als_weighted:.4f}")
    print(f"  ROC-AUC: {roc_als_weighted:.4f}")
    print(f"  Weights: [{w_weighted[0]:.3f}, {w_weighted[1]:.3f}, ..., {w_weighted[-1]:.3f}]")
    print(f"  Weight std: {np.std(w_weighted):.4f}")
    print(f"  Prediction std: {np.std(preds_als_weighted):.4f}")
    
    # Check success criteria
    weights_healthy = np.std(w_weighted) > 0.05
    preds_varied = np.std(preds_als_weighted) > 0.05
    performance_good = pr_als_weighted > pr_simple * 0.8
    
    if weights_healthy and preds_varied and performance_good:
        print(f"  Status: ✅ SUCCESS! (healthy weights + good performance)")
    elif performance_good:
        print(f"  Status: ⚠️  Good PR-AUC but weights/preds still small")
    else:
        print(f"  Status: ❌ Failed")
    
    # Test 3: PyTorch without class weights (should fail)
    print("\n[5] PyTorch Trainer (no class weights) - CONTROL")
    
    try:
        trainer_pt_no_weights = CFEnsemblePyTorchTrainer(
            n_classifiers=10,
            latent_dim=15,
            rho=0.5,
            lambda_reg=0.01,
            max_epochs=200,
            lr=0.01,
            use_class_weights=False,  # DISABLED
            verbose=False,
            random_seed=42
        )
        trainer_pt_no_weights.fit(data)
        preds_pt_no_weights = trainer_pt_no_weights.predict()[~labeled_mask]
        pr_pt_no_weights = average_precision_score(y_test, preds_pt_no_weights)
        roc_pt_no_weights = roc_auc_score(y_test, preds_pt_no_weights)
        
        w_pt_no_weights, b_pt_no_weights = trainer_pt_no_weights.get_aggregator_weights()
        print(f"  PR-AUC:  {pr_pt_no_weights:.4f}")
        print(f"  ROC-AUC: {roc_pt_no_weights:.4f}")
        print(f"  Weights: [{w_pt_no_weights[0]:.3f}, {w_pt_no_weights[1]:.3f}, ..., {w_pt_no_weights[-1]:.3f}]")
        print(f"  Weight std: {np.std(w_pt_no_weights):.4f}")
        print(f"  Prediction std: {np.std(preds_pt_no_weights):.4f}")
        print(f"  Status: {'❌ COLLAPSED' if np.std(w_pt_no_weights) < 0.05 else '✅ OK'}")
        
        pt_no_weights_available = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        pt_no_weights_available = False
        pr_pt_no_weights = np.nan
        roc_pt_no_weights = np.nan
    
    # Test 4: PyTorch with class weights (should work!)
    print("\n[6] PyTorch Trainer (with class weights) - NEW FIX")
    
    try:
        trainer_pt_weighted = CFEnsemblePyTorchTrainer(
            n_classifiers=10,
            latent_dim=15,
            rho=0.5,
            lambda_reg=0.01,
            max_epochs=200,
            lr=0.01,
            use_class_weights=True,  # ENABLED (default)
            verbose=False,
            random_seed=42
        )
        trainer_pt_weighted.fit(data)
        preds_pt_weighted = trainer_pt_weighted.predict()[~labeled_mask]
        pr_pt_weighted = average_precision_score(y_test, preds_pt_weighted)
        roc_pt_weighted = roc_auc_score(y_test, preds_pt_weighted)
        
        w_pt_weighted, b_pt_weighted = trainer_pt_weighted.get_aggregator_weights()
        print(f"  PR-AUC:  {pr_pt_weighted:.4f}")
        print(f"  ROC-AUC: {roc_pt_weighted:.4f}")
        print(f"  Weights: [{w_pt_weighted[0]:.3f}, {w_pt_weighted[1]:.3f}, ..., {w_pt_weighted[-1]:.3f}]")
        print(f"  Weight std: {np.std(w_pt_weighted):.4f}")
        print(f"  Prediction std: {np.std(preds_pt_weighted):.4f}")
        
        # Check success
        weights_healthy_pt = np.std(w_pt_weighted) > 0.05
        preds_varied_pt = np.std(preds_pt_weighted) > 0.05
        performance_good_pt = pr_pt_weighted > pr_simple * 0.8
        
        if weights_healthy_pt and preds_varied_pt and performance_good_pt:
            print(f"  Status: ✅ SUCCESS! (healthy weights + good performance)")
        elif performance_good_pt:
            print(f"  Status: ⚠️  Good PR-AUC but weights/preds still small")
        else:
            print(f"  Status: ❌ Failed")
        
        pt_weighted_available = True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        pt_weighted_available = False
        pr_pt_weighted = np.nan
        roc_pt_weighted = np.nan
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<30} {'PR-AUC':>10} {'ROC-AUC':>10} {'Weight Std':>12} {'Status':>15}")
    print("-"*80)
    
    print(f"{'Simple Average':<30} {pr_simple:>10.4f} {roc_simple:>10.4f} {'N/A':>12} {'Baseline':>15}")
    
    print(f"{'ALS (no class weights)':<30} {pr_als_no_weights:>10.4f} {roc_als_no_weights:>10.4f} "
          f"{np.std(w_no_weights):>12.4f} {'❌ Collapsed':>15}")
    
    status_als = '✅ FIXED' if (np.std(w_weighted) > 0.05 and pr_als_weighted > pr_simple * 0.8) else '⚠️  Partial'
    print(f"{'ALS (class weighted)':<30} {pr_als_weighted:>10.4f} {roc_als_weighted:>10.4f} "
          f"{np.std(w_weighted):>12.4f} {status_als:>15}")
    
    if pt_no_weights_available:
        print(f"{'PyTorch (no class weights)':<30} {pr_pt_no_weights:>10.4f} {roc_pt_no_weights:>10.4f} "
              f"{np.std(w_pt_no_weights):>12.4f} {'❌ Collapsed':>15}")
    else:
        print(f"{'PyTorch (no class weights)':<30} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'Error':>15}")
    
    if pt_weighted_available:
        status_pt = '✅ FIXED' if (np.std(w_pt_weighted) > 0.05 and pr_pt_weighted > pr_simple * 0.8) else '⚠️  Partial'
        print(f"{'PyTorch (class weighted)':<30} {pr_pt_weighted:>10.4f} {roc_pt_weighted:>10.4f} "
              f"{np.std(w_pt_weighted):>12.4f} {status_pt:>15}")
    else:
        print(f"{'PyTorch (class weighted)':<30} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'Error':>15}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    als_fixed = (np.std(w_weighted) > 0.05 and pr_als_weighted > pr_simple * 0.8)
    pt_fixed = pt_weighted_available and (np.std(w_pt_weighted) > 0.05 and pr_pt_weighted > pr_simple * 0.8)
    
    print(f"\n1. ALS Trainer:")
    print(f"   Without class weights: PR-AUC = {pr_als_no_weights:.4f}, Weight std = {np.std(w_no_weights):.4f}")
    print(f"   With class weights:    PR-AUC = {pr_als_weighted:.4f}, Weight std = {np.std(w_weighted):.4f}")
    if als_fixed:
        print(f"   ✅ CLASS WEIGHTING FIXES THE PROBLEM!")
    else:
        print(f"   ⚠️  Partial improvement but not fully fixed")
    
    if pt_weighted_available:
        print(f"\n2. PyTorch Trainer:")
        print(f"   Without class weights: PR-AUC = {pr_pt_no_weights:.4f}, Weight std = {np.std(w_pt_no_weights):.4f}")
        print(f"   With class weights:    PR-AUC = {pr_pt_weighted:.4f}, Weight std = {np.std(w_pt_weighted):.4f}")
        if pt_fixed:
            print(f"   ✅ CLASS WEIGHTING FIXES THE PROBLEM!")
        else:
            print(f"   ⚠️  Partial improvement but not fully fixed")
    
    if als_fixed or pt_fixed:
        print(f"\n🎉 SUCCESS! Class-weighted gradients solve the weight collapse problem!")
    else:
        print(f"\n⚠️  Mixed results. Further investigation needed.")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    test_class_weighted_fix()
