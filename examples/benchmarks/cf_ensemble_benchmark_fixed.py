"""
Benchmark CF-Ensemble vs. Baselines on Imbalanced Data (FIXED)

CRITICAL FIX: Proper transductive learning
- Train on ALL data (train + test) with test labels masked
- Use learned latent factors for prediction (not cold-start inductive)

Tests the key hypothesis: Does CF transformation improve ensemble performance,
especially on class-imbalanced datasets?

Comparison methods:
1. Simple averaging (baseline)
2. Weighted averaging (certainty-based)
3. Stacking (logistic regression)
4. CF-Ensemble with different ρ values

Metrics: PR-AUC (primary), ROC-AUC, F1-score for imbalanced data
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
    precision_recall_curve
)

# Import CF-Ensemble components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import EnsembleData, generate_imbalanced_ensemble_data
from cfensemble.optimization import CFEnsembleTrainer


def simple_average(R: np.ndarray) -> np.ndarray:
    """Baseline: Simple average of all classifiers."""
    return np.mean(R, axis=0)


def weighted_average_certainty(R: np.ndarray) -> np.ndarray:
    """Weighted average using certainty as weights."""
    # Weight by certainty: |r - 0.5|
    certainty = np.abs(R - 0.5)
    weights = certainty / (certainty.sum(axis=0, keepdims=True) + 1e-10)
    return np.sum(weights * R, axis=0)


def stacking_logreg(R_train: np.ndarray, y_train: np.ndarray, 
                     R_test: np.ndarray) -> np.ndarray:
    """Stacking with logistic regression."""
    # Train meta-learner on training predictions
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(R_train.T, y_train)  # Transpose to (n_samples, n_classifiers)
    
    # Predict on test
    y_pred_proba = model.predict_proba(R_test.T)[:, 1]
    return y_pred_proba


def cf_ensemble_method(R_train: np.ndarray, y_train: np.ndarray,
                       R_test: np.ndarray, rho: float = 0.5,
                       latent_dim: int = 20, lambda_reg: float = 0.01,
                       max_iter: int = 200) -> Tuple[np.ndarray, Dict]:
    """
    CF-Ensemble transformation with PROPER transductive learning.
    
    FIXED: Uses learned latent factors for test predictions (transductive),
    not cold-start inductive prediction.
    
    Parameters:
    -----------
    R_train : array (m, n_train)
        Training predictions
    y_train : array (n_train,)
        Training labels
    R_test : array (m, n_test)
        Test predictions
    rho : float
        Trade-off: 1.0=pure reconstruction, 0.0=pure supervised
    latent_dim : int
        Latent factor dimension (increased from 10 to 20)
    lambda_reg : float
        Regularization (decreased from 0.1 to 0.01)
    max_iter : int
        Max iterations (increased from 100 to 200)
        
    Returns:
    --------
    y_pred : array (n_test,)
        Predictions on test set
    info : dict
        Additional information (losses, factors, etc.)
    """
    m, n_train = R_train.shape
    n_test = R_test.shape[1]
    
    # CRITICAL: Combine train and test, mask test labels
    # This is the CORRECT way for transductive learning
    R_combined = np.hstack([R_train, R_test])
    labels_combined = np.concatenate([
        y_train,
        np.full(n_test, np.nan)  # Masked test labels
    ])
    
    # Create ensemble data
    ensemble_data = EnsembleData(R_combined, labels_combined)
    
    # Train CF-Ensemble on ALL data (train + test)
    trainer = CFEnsembleTrainer(
        n_classifiers=m,
        latent_dim=latent_dim,
        rho=rho,
        lambda_reg=lambda_reg,
        max_iter=max_iter,
        aggregator_lr=0.1,  # Increased from 0.01
        verbose=False
    )
    
    trainer.fit(ensemble_data)
    
    # FIXED: Get predictions using learned latent factors (transductive)
    # NOT using R_new (which does cold-start inductive prediction)
    all_predictions = trainer.predict()  # Uses learned Y factors
    y_pred = all_predictions[n_train:]  # Extract test portion
    
    # Collect info
    info = {
        'final_loss': trainer.history['loss'][-1] if trainer.history['loss'] else np.nan,
        'recon_loss': trainer.history['reconstruction'][-1] if trainer.history['reconstruction'] else np.nan,
        'sup_loss': trainer.history['supervised'][-1] if trainer.history['supervised'] else np.nan,
        'n_iter': trainer.n_iter_,
        'converged': trainer.converged_,
        'rho': rho,
        'latent_dim': latent_dim,
        'lambda_reg': lambda_reg
    }
    
    return y_pred, info


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for imbalanced data."""
    # Clip predictions to valid probability range
    y_pred_clipped = np.clip(y_pred, 0, 1)
    
    # Binary predictions (threshold at 0.5)
    y_pred_binary = (y_pred_clipped >= 0.5).astype(int)
    
    metrics = {
        'pr_auc': average_precision_score(y_true, y_pred_clipped),
        'roc_auc': roc_auc_score(y_true, y_pred_clipped),
        'f1_score': f1_score(y_true, y_pred_binary),
    }
    
    return metrics


def run_single_experiment(
    positive_rate: float = 0.05,
    n_instances: int = 1000,
    n_classifiers: int = 15,
    target_quality: float = 0.70,
    latent_dim: int = 20,
    lambda_reg: float = 0.01,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run a single experiment comparing all methods.
    
    Parameters:
    -----------
    positive_rate : float
        Fraction of positive examples (class imbalance)
    n_instances : int
        Total number of instances
    n_classifiers : int
        Number of base classifiers
    target_quality : float
        Target AUC for synthetic classifiers
    latent_dim : int
        Latent dimension for CF-Ensemble
    lambda_reg : float
        Regularization strength
    seed : int
        Random seed
        
    Returns:
    --------
    results_df : DataFrame
        Results for all methods
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {positive_rate*100:.0f}% positive rate")
    print(f"{'='*60}")
    
    # Generate synthetic ensemble data
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        n_instances=n_instances,
        n_classifiers=n_classifiers,
        n_labeled=n_instances // 2,  # 50% labeled (train), 50% unlabeled (test)
        positive_rate=positive_rate,
        target_quality=target_quality,
        diversity='high',
        random_state=seed
    )
    
    # Split train/test
    # labeled_idx is boolean mask
    R_train = R[:, labeled_idx]
    R_test = R[:, ~labeled_idx]
    y_train = labels[labeled_idx]
    y_test = y_true[~labeled_idx]
    
    print(f"Data: {R_train.shape[1]} train, {R_test.shape[1]} test")
    print(f"Train labels: {np.sum(y_train == 1)} pos, {np.sum(y_train == 0)} neg")
    print(f"Test labels: {np.sum(y_test == 1)} pos, {np.sum(y_test == 0)} neg")
    
    results = []
    
    # Method 1: Simple Average
    print("\n[1/7] Simple Average...")
    y_pred = simple_average(R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    results.append({
        'method': 'Simple Average',
        'rho': np.nan,
        **metrics
    })
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Method 2: Weighted Average (Certainty)
    print("[2/7] Weighted Average (Certainty)...")
    y_pred = weighted_average_certainty(R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    results.append({
        'method': 'Weighted Average',
        'rho': np.nan,
        **metrics
    })
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Method 3: Stacking (Logistic Regression)
    print("[3/7] Stacking (Logistic Regression)...")
    y_pred = stacking_logreg(R_train, y_train, R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    results.append({
        'method': 'Stacking',
        'rho': np.nan,
        **metrics
    })
    print(f"  PR-AUC: {metrics['pr_auc']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Method 4-7: CF-Ensemble with different ρ values
    for idx, rho in enumerate([1.0, 0.7, 0.5, 0.0], start=4):
        print(f"[{idx}/7] CF-Ensemble (ρ={rho})...")
        try:
            y_pred, info = cf_ensemble_method(
                R_train, y_train, R_test,
                rho=rho,
                latent_dim=latent_dim,
                lambda_reg=lambda_reg,
                max_iter=200
            )
            metrics = evaluate_predictions(y_test, y_pred)
            
            # Print convergence info
            converged_str = "✓" if info['converged'] else "✗"
            print(f"  PR-AUC: {metrics['pr_auc']:.3f}, "
                  f"ROC-AUC: {metrics['roc_auc']:.3f}, "
                  f"Converged: {converged_str} ({info['n_iter']} iter), "
                  f"Loss: {info['final_loss']:.4f}")
            
            results.append({
                'method': f'CF-Ensemble (ρ={rho})',
                'rho': rho,
                **metrics,
                **info
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'method': f'CF-Ensemble (ρ={rho})',
                'rho': rho,
                'pr_auc': 0.0,
                'roc_auc': 0.5,
                'f1_score': 0.0,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def run_imbalance_sweep(
    positive_rates: list = [0.10, 0.05, 0.01],
    n_instances: int = 1000,
    n_classifiers: int = 15,
    latent_dim: int = 20,
    lambda_reg: float = 0.01,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run experiments across different imbalance levels.
    """
    all_results = []
    
    for pos_rate in positive_rates:
        print(f"\n{'#'*70}")
        print(f"# Running experiments with {pos_rate*100:.0f}% minority class")
        print(f"{'#'*70}")
        
        df = run_single_experiment(
            positive_rate=pos_rate,
            n_instances=n_instances,
            n_classifiers=n_classifiers,
            latent_dim=latent_dim,
            lambda_reg=lambda_reg,
            seed=seed
        )
        df['positive_rate'] = pos_rate
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True)


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get unique positive rates
    pos_rates = sorted(results_df['positive_rate'].unique())
    
    # Prepare data for plotting
    methods = results_df['method'].unique()
    x = np.arange(len(pos_rates))
    width = 0.12
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        pr_aucs = [method_data[method_data['positive_rate'] == pr]['pr_auc'].values[0] 
                   for pr in pos_rates]
        roc_aucs = [method_data[method_data['positive_rate'] == pr]['roc_auc'].values[0] 
                    for pr in pos_rates]
        f1s = [method_data[method_data['positive_rate'] == pr]['f1_score'].values[0] 
               for pr in pos_rates]
        
        offset = (i - len(methods)/2) * width
        axes[0].bar(x + offset, pr_aucs, width, label=method, alpha=0.8)
        axes[1].bar(x + offset, roc_aucs, width, label=method, alpha=0.8)
        axes[2].bar(x + offset, f1s, width, label=method, alpha=0.8)
    
    # Configure plots
    for ax, title in zip(axes, ['PR-AUC', 'ROC-AUC', 'F1-Score']):
        ax.set_xlabel('Minority Class %')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs. Class Imbalance')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pr*100:.0f}%' for pr in pos_rates])
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison_fixed.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {output_dir / 'benchmark_comparison_fixed.png'}")
    plt.close()


def main():
    """Run benchmark experiments."""
    # Setup
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'cf_benchmark_fixed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CF-ENSEMBLE BENCHMARK (FIXED - TRANSDUCTIVE LEARNING)")
    print("="*70)
    print("\nCRITICAL FIX: Proper transductive learning")
    print("  - Train on ALL data (train + test) with test labels masked")
    print("  - Use learned latent factors for prediction (not cold-start)")
    print("\nImproved hyperparameters:")
    print("  - latent_dim: 20 (increased from 10)")
    print("  - lambda_reg: 0.01 (decreased from 0.1)")
    print("  - max_iter: 200 (increased from 100)")
    print("  - aggregator_lr: 0.1 (increased from 0.01)")
    
    # Run experiments
    results_df = run_imbalance_sweep(
        positive_rates=[0.10, 0.05, 0.01],
        n_instances=1000,
        n_classifiers=15,
        latent_dim=20,
        lambda_reg=0.01,
        seed=42
    )
    
    # Save results
    results_df.to_csv(output_dir / 'raw_results_fixed.csv', index=False)
    print(f"\nResults saved: {output_dir / 'raw_results_fixed.csv'}")
    
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
    summary.to_csv(output_dir / 'summary_fixed.csv')
    
    # Plot results
    plot_results(results_df, output_dir)
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)
    print(f"Results directory: {output_dir}")


if __name__ == '__main__':
    main()
