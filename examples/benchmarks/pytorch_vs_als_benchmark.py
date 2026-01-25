"""
Benchmark: PyTorch Joint Optimization vs. ALS Alternating Optimization

Tests whether joint gradient descent via PyTorch solves the optimization
instability problem observed in the ALS-based trainer.

Comparison:
1. Simple Average (baseline)
2. Stacking (baseline)
3. CF-Ensemble with ALS (alternating optimization - may fail)
4. CF-Ensemble with PyTorch (joint optimization - should work)

Based on: docs/failure_modes/optimization_instability.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
)

# Import CF-Ensemble components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import EnsembleData, generate_imbalanced_ensemble_data
from cfensemble.optimization import CFEnsembleTrainer, PYTORCH_AVAILABLE

if PYTORCH_AVAILABLE:
    from cfensemble.optimization import CFEnsemblePyTorchTrainer
else:
    print("WARNING: PyTorch not available. Install with: pip install torch")
    print("Only ALS-based trainer will be tested.\n")


def simple_average(R: np.ndarray) -> np.ndarray:
    """Baseline: Simple average of all classifiers."""
    return np.mean(R, axis=0)


def stacking_logreg(R_train: np.ndarray, y_train: np.ndarray, 
                     R_test: np.ndarray) -> np.ndarray:
    """Stacking with logistic regression."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(R_train.T, y_train)
    y_pred_proba = model.predict_proba(R_test.T)[:, 1]
    return y_pred_proba


def cf_als_method(
    R_train: np.ndarray,
    y_train: np.ndarray,
    R_test: np.ndarray,
    rho: float = 0.5
) -> Tuple[np.ndarray, Dict]:
    """CF-Ensemble with ALS (alternating optimization)."""
    m, n_train = R_train.shape
    n_test = R_test.shape[1]
    
    # Transductive learning: combine train+test, mask test labels
    R_combined = np.hstack([R_train, R_test])
    labels_combined = np.concatenate([y_train, np.full(n_test, np.nan)])
    ensemble_data = EnsembleData(R_combined, labels_combined)
    
    trainer = CFEnsembleTrainer(
        n_classifiers=m,
        latent_dim=20,
        rho=rho,
        lambda_reg=0.01,
        max_iter=200,
        aggregator_lr=0.1,
        use_label_aware_confidence=True,  # CRITICAL: Enable approximation
        label_aware_alpha=1.0,
        verbose=False
    )
    
    trainer.fit(ensemble_data)
    
    # Transductive prediction
    all_preds = trainer.predict()
    y_pred = all_preds[n_train:]
    
    info = {
        'converged': trainer.converged_,
        'n_iter': trainer.n_iter_,
        'final_loss': trainer.history['loss'][-1] if trainer.history['loss'] else np.nan,
        'recon_loss': trainer.history['reconstruction'][-1] if trainer.history['reconstruction'] else np.nan,
        'sup_loss': trainer.history['supervised'][-1] if trainer.history['supervised'] else np.nan,
    }
    
    return y_pred, info


def cf_pytorch_method(
    R_train: np.ndarray,
    y_train: np.ndarray,
    R_test: np.ndarray,
    rho: float = 0.5
) -> Tuple[np.ndarray, Dict]:
    """CF-Ensemble with PyTorch (joint optimization)."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    m, n_train = R_train.shape
    n_test = R_test.shape[1]
    
    # Transductive learning: combine train+test, mask test labels
    R_combined = np.hstack([R_train, R_test])
    labels_combined = np.concatenate([y_train, np.full(n_test, np.nan)])
    ensemble_data = EnsembleData(R_combined, labels_combined)
    
    trainer = CFEnsemblePyTorchTrainer(
        n_classifiers=m,
        latent_dim=20,
        rho=rho,
        lambda_reg=0.01,
        max_epochs=200,
        lr=0.01,
        optimizer='adam',
        patience=20,
        verbose=False
    )
    
    trainer.fit(ensemble_data)
    
    # Transductive prediction
    all_preds = trainer.predict()
    y_pred = all_preds[n_train:]
    
    info = {
        'converged': trainer.converged_,
        'n_epochs': trainer.n_epochs_,
        'final_loss': trainer.history['loss'][-1] if trainer.history['loss'] else np.nan,
        'recon_loss': trainer.history['reconstruction'][-1] if trainer.history['reconstruction'] else np.nan,
        'sup_loss': trainer.history['supervised'][-1] if trainer.history['supervised'] else np.nan,
    }
    
    return y_pred, info


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    y_pred_clipped = np.clip(y_pred, 0, 1)
    y_pred_binary = (y_pred_clipped >= 0.5).astype(int)
    
    metrics = {
        'pr_auc': average_precision_score(y_true, y_pred_clipped),
        'roc_auc': roc_auc_score(y_true, y_pred_clipped),
        'f1_score': f1_score(y_true, y_pred_binary),
    }
    
    return metrics


def run_single_experiment(
    positive_rate: float = 0.10,
    n_instances: int = 1000,
    n_classifiers: int = 15,
    target_quality: float = 0.70,
    seed: int = 42
) -> pd.DataFrame:
    """Run a single experiment comparing all methods."""
    print(f"\n{'='*70}")
    print(f"Experiment: {positive_rate*100:.0f}% positive rate")
    print(f"{'='*70}")
    
    # Generate data
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        n_instances=n_instances,
        n_classifiers=n_classifiers,
        n_labeled=n_instances // 2,
        positive_rate=positive_rate,
        target_quality=target_quality,
        diversity='high',
        random_state=seed
    )
    
    # Convert to boolean mask
    n = R.shape[1]
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    # Split train/test
    R_train = R[:, labeled_mask]
    R_test = R[:, ~labeled_mask]
    y_train = labels[labeled_mask]
    y_test = y_true[~labeled_mask]
    
    print(f"Data: {R_train.shape[1]} train, {R_test.shape[1]} test")
    print(f"Positive: {np.sum(y_test == 1)} / {len(y_test)}")
    
    results = []
    
    # Method 1: Simple Average
    print("[1/6] Simple Average...")
    y_pred = simple_average(R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    results.append({'method': 'Simple Average', **metrics})
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
    
    # Method 2: Stacking
    print("[2/6] Stacking...")
    y_pred = stacking_logreg(R_train, y_train, R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    results.append({'method': 'Stacking', **metrics})
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
    
    # Method 3-4: CF-ALS with different ρ
    for idx, rho in enumerate([0.5, 0.0], start=3):
        print(f"[{idx}/6] CF-Ensemble ALS (ρ={rho})...")
        try:
            y_pred, info = cf_als_method(R_train, y_train, R_test, rho=rho)
            metrics = evaluate_predictions(y_test, y_pred)
            
            converged_str = "✓" if info['converged'] else "✗"
            print(f"  PR-AUC: {metrics['pr_auc']:.3f}, Converged: {converged_str} ({info['n_iter']} iter)")
            
            results.append({
                'method': f'CF-ALS (ρ={rho})',
                **metrics,
                **info
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'method': f'CF-ALS (ρ={rho})',
                'pr_auc': 0.0,
                'roc_auc': 0.5,
                'f1_score': 0.0,
                'error': str(e)
            })
    
    # Method 5-6: CF-PyTorch with different ρ
    if PYTORCH_AVAILABLE:
        for idx, rho in enumerate([0.5, 0.0], start=5):
            print(f"[{idx}/6] CF-Ensemble PyTorch (ρ={rho})...")
            try:
                y_pred, info = cf_pytorch_method(R_train, y_train, R_test, rho=rho)
                metrics = evaluate_predictions(y_test, y_pred)
                
                converged_str = "✓" if info['converged'] else "✗"
                print(f"  PR-AUC: {metrics['pr_auc']:.3f}, Converged: {converged_str} ({info['n_epochs']} epochs)")
                
                results.append({
                    'method': f'CF-PyTorch (ρ={rho})',
                    **metrics,
                    **info
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'method': f'CF-PyTorch (ρ={rho})',
                    'pr_auc': 0.0,
                    'roc_auc': 0.5,
                    'f1_score': 0.0,
                    'error': str(e)
                })
    else:
        print("[5-6/6] CF-PyTorch: SKIPPED (PyTorch not available)")
    
    df = pd.DataFrame(results)
    df['positive_rate'] = positive_rate
    return df


def run_benchmark(
    positive_rates: list = [0.10, 0.05, 0.01],
    seed: int = 42
) -> pd.DataFrame:
    """Run experiments across different imbalance levels."""
    all_results = []
    
    for pos_rate in positive_rates:
        df = run_single_experiment(
            positive_rate=pos_rate,
            seed=seed
        )
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pos_rates = sorted(results_df['positive_rate'].unique())
    methods = results_df['method'].unique()
    x = np.arange(len(pos_rates))
    width = 0.13
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        pr_aucs = [method_data[method_data['positive_rate'] == pr]['pr_auc'].values[0] 
                   for pr in pos_rates]
        
        offset = (i - len(methods)/2) * width
        axes[0].bar(x + offset, pr_aucs, width, label=method, alpha=0.8)
    
    axes[0].set_xlabel('Minority Class %')
    axes[0].set_ylabel('PR-AUC')
    axes[0].set_title('Performance Comparison: PR-AUC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{pr*100:.0f}%' for pr in pos_rates])
    axes[0].legend(fontsize=8, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Convergence comparison (if available)
    convergence_data = []
    for method in methods:
        if 'CF-' in method:
            method_subset = results_df[results_df['method'] == method]
            converged_count = method_subset.get('converged', pd.Series([False])).sum()
            total = len(method_subset)
            convergence_data.append({
                'method': method,
                'convergence_rate': converged_count / total if total > 0 else 0
            })
    
    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)
        axes[1].barh(conv_df['method'], conv_df['convergence_rate'])
        axes[1].set_xlabel('Convergence Rate')
        axes[1].set_title('Optimization Convergence')
        axes[1].set_xlim([0, 1])
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No convergence data', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pytorch_vs_als_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_dir / 'pytorch_vs_als_comparison.png'}")
    plt.close()


def main():
    """Run benchmark experiments."""
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'pytorch_vs_als'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CF-ENSEMBLE BENCHMARK: PyTorch vs. ALS")
    print("="*70)
    print("\nGoal: Test if PyTorch joint optimization solves the instability")
    print("      problem observed in ALS alternating optimization.\n")
    
    if PYTORCH_AVAILABLE:
        print("✓ PyTorch available - will compare both methods")
    else:
        print("✗ PyTorch NOT available - only testing ALS")
        print("  Install with: pip install torch\n")
    
    # Run experiments
    results_df = run_benchmark(
        positive_rates=[0.10, 0.05, 0.01],
        seed=42
    )
    
    # Save results
    results_df.to_csv(output_dir / 'results.csv', index=False)
    print(f"\nResults saved: {output_dir / 'results.csv'}")
    
    # Create summary
    summary = results_df.pivot_table(
        index='method',
        columns='positive_rate',
        values='pr_auc',
        aggfunc='mean'
    )
    summary.columns = [f'{c*100:.0f}% pos' for c in summary.columns]
    
    print("\n" + "="*70)
    print("SUMMARY: PR-AUC by Method and Imbalance Level")
    print("="*70)
    print(summary.round(3))
    
    summary.to_csv(output_dir / 'summary.csv')
    
    # Plot results
    plot_results(results_df, output_dir)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    simple_avg = results_df[results_df['method'] == 'Simple Average']['pr_auc'].mean()
    stacking = results_df[results_df['method'] == 'Stacking']['pr_auc'].mean()
    
    print(f"\nBaselines:")
    print(f"  Simple Average: {simple_avg:.3f}")
    print(f"  Stacking: {stacking:.3f}")
    
    als_methods = results_df[results_df['method'].str.contains('CF-ALS')]
    if not als_methods.empty:
        als_avg = als_methods['pr_auc'].mean()
        als_converged = als_methods['converged'].sum() if 'converged' in als_methods.columns else 0
        als_total = len(als_methods)
        print(f"\nCF-ALS:")
        print(f"  Average PR-AUC: {als_avg:.3f}")
        print(f"  Convergence: {als_converged}/{als_total} ({als_converged/als_total*100:.0f}%)")
        
        if als_avg < simple_avg:
            print(f"  ❌ WORSE than simple average ({als_avg:.3f} < {simple_avg:.3f})")
        elif als_avg < stacking:
            print(f"  ⚠️  Better than simple avg, worse than stacking")
        else:
            print(f"  ✓ BEST method!")
    
    pytorch_methods = results_df[results_df['method'].str.contains('CF-PyTorch')]
    if not pytorch_methods.empty:
        pytorch_avg = pytorch_methods['pr_auc'].mean()
        pytorch_converged = pytorch_methods['converged'].sum() if 'converged' in pytorch_methods.columns else 0
        pytorch_total = len(pytorch_methods)
        print(f"\nCF-PyTorch:")
        print(f"  Average PR-AUC: {pytorch_avg:.3f}")
        print(f"  Convergence: {pytorch_converged}/{pytorch_total} ({pytorch_converged/pytorch_total*100:.0f}%)")
        
        if pytorch_avg < simple_avg:
            print(f"  ❌ WORSE than simple average ({pytorch_avg:.3f} < {simple_avg:.3f})")
        elif pytorch_avg < stacking:
            print(f"  ⚠️  Better than simple avg, worse than stacking")
        else:
            print(f"  ✓ BEST method!")
        
        if not als_methods.empty:
            improvement = pytorch_avg - als_avg
            print(f"\nPyTorch vs. ALS:")
            if improvement > 0.01:
                print(f"  ✓ PyTorch is BETTER (+{improvement:.3f} PR-AUC)")
                print(f"  Joint optimization SOLVES the instability problem!")
            elif improvement < -0.01:
                print(f"  ✗ PyTorch is WORSE ({improvement:.3f} PR-AUC)")
            else:
                print(f"  ≈ Similar performance ({improvement:+.3f} PR-AUC)")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)
    print(f"Results directory: {output_dir}")


if __name__ == '__main__':
    main()
