"""
Diagnose Synthetic Data Quality

Checks if generate_imbalanced_ensemble_data actually achieves target quality.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import generate_imbalanced_ensemble_data
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def diagnose_data_quality(
    target_quality=0.70,
    positive_rate=0.10,
    n_trials=5,
    random_state_base=42
):
    """
    Generate data multiple times and check actual achieved quality.
    """
    print("="*70)
    print(f"DIAGNOSTIC: Synthetic Data Quality")
    print(f"Target Quality: {target_quality:.2f}")
    print(f"Positive Rate: {positive_rate:.2%}")
    print("="*70)
    
    all_accuracies = []
    all_roc_aucs = []
    all_pr_aucs = []
    
    for trial in range(n_trials):
        R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
            n_instances=1000,
            n_classifiers=15,
            n_labeled=500,
            positive_rate=positive_rate,
            target_quality=target_quality,
            diversity='high',
            random_state=random_state_base + trial
        )
        
        # Measure quality of each classifier on ALL data
        m, n = R.shape
        
        trial_accs = []
        trial_roc = []
        trial_pr = []
        
        for u in range(m):
            # Accuracy
            preds_binary = (R[u, :] > 0.5).astype(float)
            acc = accuracy_score(y_true, preds_binary)
            trial_accs.append(acc)
            
            # ROC-AUC
            roc = roc_auc_score(y_true, R[u, :])
            trial_roc.append(roc)
            
            # PR-AUC
            pr = average_precision_score(y_true, R[u, :])
            trial_pr.append(pr)
        
        all_accuracies.extend(trial_accs)
        all_roc_aucs.extend(trial_roc)
        all_pr_aucs.extend(trial_pr)
        
        print(f"\nTrial {trial+1}:")
        print(f"  Accuracy:  {np.mean(trial_accs):.3f} ± {np.std(trial_accs):.3f}  (range: [{np.min(trial_accs):.3f}, {np.max(trial_accs):.3f}])")
        print(f"  ROC-AUC:   {np.mean(trial_roc):.3f} ± {np.std(trial_roc):.3f}  (range: [{np.min(trial_roc):.3f}, {np.max(trial_roc):.3f}])")
        print(f"  PR-AUC:    {np.mean(trial_pr):.3f} ± {np.std(trial_pr):.3f}  (range: [{np.min(trial_pr):.3f}, {np.max(trial_pr):.3f}])")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS (across all trials):")
    print("="*70)
    print(f"Accuracy:  {np.mean(all_accuracies):.3f} ± {np.std(all_accuracies):.3f}")
    print(f"ROC-AUC:   {np.mean(all_roc_aucs):.3f} ± {np.std(all_roc_aucs):.3f}")
    print(f"PR-AUC:    {np.mean(all_pr_aucs):.3f} ± {np.std(all_pr_aucs):.3f}")
    
    # Compare to target
    print("\n" + "="*70)
    print("COMPARISON TO TARGET:")
    print("="*70)
    
    if positive_rate <= 0.2:
        # For imbalanced, use PR-AUC
        achieved = np.mean(all_pr_aucs)
        metric = "PR-AUC"
    else:
        # For balanced, use ROC-AUC
        achieved = np.mean(all_roc_aucs)
        metric = "ROC-AUC"
    
    print(f"Target {metric}: {target_quality:.3f}")
    print(f"Achieved {metric}: {achieved:.3f}")
    print(f"Gap: {target_quality - achieved:.3f} ({(target_quality - achieved)/target_quality * 100:.1f}%)")
    
    if achieved < target_quality - 0.05:
        print("\n❌ FAIL: Achieved quality significantly below target!")
        print("   Synthetic data generation needs fixing.")
    elif achieved < target_quality:
        print("\n⚠️  WARNING: Achieved quality slightly below target.")
        print("   Within acceptable range, but could be improved.")
    else:
        print("\n✅ PASS: Achieved quality meets or exceeds target!")
    
    return achieved, target_quality


def test_different_targets():
    """Test multiple target quality levels."""
    print("\n" + "#"*70)
    print("# Testing Multiple Target Quality Levels")
    print("#"*70)
    
    results = []
    
    for target in [0.60, 0.70, 0.80]:
        print(f"\n\n{'='*70}")
        print(f"TARGET QUALITY = {target:.2f}")
        print(f"{'='*70}")
        
        achieved, target_val = diagnose_data_quality(
            target_quality=target,
            positive_rate=0.10,
            n_trials=3
        )
        
        results.append({
            'target': target,
            'achieved': achieved,
            'gap': target - achieved
        })
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: All Target Levels")
    print("="*70)
    print(f"{'Target':<10} {'Achieved':<10} {'Gap':<10} {'Status'}")
    print("-"*70)
    
    for r in results:
        status = "✓" if r['gap'] < 0.05 else "✗"
        print(f"{r['target']:<10.2f} {r['achieved']:<10.3f} {r['gap']:<10.3f} {status}")


if __name__ == '__main__':
    # Test single target
    diagnose_data_quality(target_quality=0.70, positive_rate=0.10, n_trials=5)
    
    # Test multiple targets
    test_different_targets()
