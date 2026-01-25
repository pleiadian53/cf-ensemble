"""
Benchmark CF-Ensemble vs. Baselines on Imbalanced Data

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
                       latent_dim: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    CF-Ensemble transformation.
    
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
        Latent factor dimension
        
    Returns:
    --------
    y_pred : array (n_test,)
        Predictions on test set
    info : dict
        Additional information (losses, factors, etc.)
    """
    m, n_train = R_train.shape
    n_test = R_test.shape[1]
    
    # Combine train and test (labels are NaN for test)
    R_combined = np.hstack([R_train, R_test])
    labels_combined = np.concatenate([
        y_train,
        np.full(n_test, np.nan)
    ])
    
    # Create ensemble data
    ensemble_data = EnsembleData(R_combined, labels_combined)
    
    # Train CF-Ensemble
    trainer = CFEnsembleTrainer(
        n_classifiers=m,
        latent_dim=latent_dim,
        rho=rho,
        lambda_reg=0.1,
        max_iter=100,
        verbose=False  # Reduce output
    )
    
    trainer.fit(ensemble_data)
    
    # Get predictions on test set (inductive)
    y_pred = trainer.predict(R_new=R_test)
    
    # Collect info
    info = {
        'final_loss': trainer.history['loss'][-1] if trainer.history['loss'] else np.nan,
        'n_iter': trainer.n_iter_,
        'converged': trainer.converged_,
        'rho': rho
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
    latent_dim: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run a single benchmark experiment.
    
    Parameters:
    -----------
    positive_rate : float
        Minority class rate (e.g., 0.05 = 5% positives)
    n_instances : int
        Total number of instances
    n_classifiers : int
        Number of base classifiers
    target_quality : float
        Target quality (ROC-AUC) for base classifiers
    latent_dim : int
        Latent factor dimension for CF-Ensemble
    seed : int
        Random seed
        
    Returns:
    --------
    results_df : DataFrame
        Results for all methods
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {positive_rate*100:.1f}% minority class")
    print(f"{'='*70}")
    
    # Generate synthetic imbalanced data
    print("Generating synthetic ensemble data...")
    n_labeled = n_instances // 2  # 50% labeled (train), 50% unlabeled (test)
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        positive_rate=positive_rate,
        n_instances=n_instances,
        n_labeled=n_labeled,
        n_classifiers=n_classifiers,
        target_quality=target_quality,
        diversity='high',
        random_state=seed
    )
    
    print(f"Data shape: R={R.shape}, labels={labels.shape}")
    
    # Convert labeled_idx to boolean mask if it's indices
    if labeled_idx.dtype != bool:
        labeled_mask = np.zeros(len(labels), dtype=bool)
        labeled_mask[labeled_idx] = True
        labeled_idx = labeled_mask
    
    print(f"Labeled: {labeled_idx.sum()}, Unlabeled: {(~labeled_idx).sum()}")
    print(f"Positive rate: {y_true.mean():.3f}")
    
    # Split train/test
    R_train = R[:, labeled_idx]
    y_train = y_true[labeled_idx]
    R_test = R[:, ~labeled_idx]
    y_test = y_true[~labeled_idx]
    
    print(f"Train: {R_train.shape[1]}, Test: {R_test.shape[1]}")
    
    # Store results
    results = []
    
    # ========================================
    # Method 1: Simple Average
    # ========================================
    print("\n[1/6] Simple average...")
    y_pred = simple_average(R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['method'] = 'Simple Average'
    metrics['rho'] = np.nan
    results.append(metrics)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # ========================================
    # Method 2: Weighted Average (Certainty)
    # ========================================
    print("\n[2/6] Weighted average (certainty)...")
    y_pred = weighted_average_certainty(R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['method'] = 'Weighted Avg (Certainty)'
    metrics['rho'] = np.nan
    results.append(metrics)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # ========================================
    # Method 3: Stacking (Logistic Regression)
    # ========================================
    print("\n[3/6] Stacking (logistic regression)...")
    y_pred = stacking_logreg(R_train, y_train, R_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['method'] = 'Stacking (LogReg)'
    metrics['rho'] = np.nan
    results.append(metrics)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # ========================================
    # Method 4: CF-Ensemble (ρ=1.0, pure reconstruction)
    # ========================================
    print("\n[4/6] CF-Ensemble (ρ=1.0, pure reconstruction)...")
    y_pred, info = cf_ensemble_method(R_train, y_train, R_test, rho=1.0, latent_dim=latent_dim)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['method'] = 'CF-Ensemble'
    metrics['rho'] = 1.0
    results.append(metrics)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Converged: {info['converged']}, Iterations: {info['n_iter']}")
    
    # ========================================
    # Method 5: CF-Ensemble (ρ=0.5, balanced)
    # ========================================
    print("\n[5/6] CF-Ensemble (ρ=0.5, balanced objective)...")
    y_pred, info = cf_ensemble_method(R_train, y_train, R_test, rho=0.5, latent_dim=latent_dim)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['method'] = 'CF-Ensemble'
    metrics['rho'] = 0.5
    results.append(metrics)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Converged: {info['converged']}, Iterations: {info['n_iter']}")
    
    # ========================================
    # Method 6: CF-Ensemble (ρ=0.0, pure supervised)
    # ========================================
    print("\n[6/6] CF-Ensemble (ρ=0.0, pure supervised)...")
    y_pred, info = cf_ensemble_method(R_train, y_train, R_test, rho=0.0, latent_dim=latent_dim)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics['method'] = 'CF-Ensemble'
    metrics['rho'] = 0.0
    results.append(metrics)
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Converged: {info['converged']}, Iterations: {info['n_iter']}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df['positive_rate'] = positive_rate
    results_df['n_classifiers'] = n_classifiers
    results_df['target_quality'] = target_quality
    
    return results_df


def run_imbalance_sweep(
    positive_rates: list = [0.10, 0.05, 0.01],
    n_trials: int = 3,
    output_dir: str = 'results/cf_benchmark'
) -> pd.DataFrame:
    """
    Run benchmarks across different imbalance levels.
    
    Parameters:
    -----------
    positive_rates : list
        Minority class rates to test
    n_trials : int
        Number of trials per configuration
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    all_results : DataFrame
        Combined results across all experiments
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for positive_rate in positive_rates:
        for trial in range(n_trials):
            seed = 42 + trial
            print(f"\n{'#'*70}")
            print(f"# Imbalance: {positive_rate*100:.1f}%, Trial: {trial+1}/{n_trials}")
            print(f"{'#'*70}")
            
            results_df = run_single_experiment(
                positive_rate=positive_rate,
                seed=seed
            )
            results_df['trial'] = trial
            
            all_results.append(results_df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save raw results
    combined_df.to_csv(output_path / 'raw_results.csv', index=False)
    print(f"\n✓ Raw results saved to: {output_path / 'raw_results.csv'}")
    
    # Compute summary statistics
    summary_df = combined_df.groupby(['positive_rate', 'method', 'rho']).agg({
        'pr_auc': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).reset_index()
    
    summary_df.to_csv(output_path / 'summary.csv', index=False)
    print(f"✓ Summary saved to: {output_path / 'summary.csv'}")
    
    return combined_df


def plot_results(results_df: pd.DataFrame, output_dir: str = 'results/cf_benchmark'):
    """Create visualization of benchmark results."""
    output_path = Path(output_dir)
    
    # Compute means across trials
    summary = results_df.groupby(['positive_rate', 'method', 'rho']).agg({
        'pr_auc': 'mean',
        'roc_auc': 'mean',
        'f1_score': 'mean'
    }).reset_index()
    
    # Create method labels
    summary['method_label'] = summary.apply(
        lambda row: f"{row['method']}" if pd.isna(row['rho']) 
        else f"{row['method']} (ρ={row['rho']:.1f})",
        axis=1
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot for each imbalance level
    for ax, pos_rate in zip(axes, sorted(summary['positive_rate'].unique())):
        data = summary[summary['positive_rate'] == pos_rate]
        
        # Sort by PR-AUC
        data = data.sort_values('pr_auc', ascending=False)
        
        # Bar plot
        x_pos = np.arange(len(data))
        ax.bar(x_pos, data['pr_auc'], alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data['method_label'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('PR-AUC', fontsize=12)
        ax.set_title(f'{pos_rate*100:.0f}% Minority Class', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.text(i, row['pr_auc'] + 0.02, f"{row['pr_auc']:.3f}",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path / 'benchmark_comparison.png'}")
    plt.close()
    
    # Print best methods
    print("\n" + "="*70)
    print("BEST METHODS BY IMBALANCE LEVEL")
    print("="*70)
    for pos_rate in sorted(summary['positive_rate'].unique()):
        data = summary[summary['positive_rate'] == pos_rate]
        best = data.loc[data['pr_auc'].idxmax()]
        print(f"\n{pos_rate*100:.0f}% Minority:")
        print(f"  Best: {best['method_label']}")
        print(f"  PR-AUC: {best['pr_auc']:.4f}")
        print(f"  ROC-AUC: {best['roc_auc']:.4f}")
        print(f"  F1: {best['f1_score']:.4f}")


def main():
    """Run complete CF-Ensemble benchmark."""
    print("\n" + "="*70)
    print("CF-ENSEMBLE BENCHMARK: Does CF Transformation Help?")
    print("="*70)
    print("\nTesting on class-imbalanced datasets")
    print("Comparing: Simple Avg, Weighted Avg, Stacking, CF-Ensemble (various ρ)")
    print("Primary metric: PR-AUC (appropriate for imbalanced data)")
    
    # Run experiments
    results_df = run_imbalance_sweep(
        positive_rates=[0.10, 0.05, 0.01],  # Test multiple imbalance levels
        n_trials=3,  # Multiple trials for robustness
        output_dir='results/cf_benchmark'
    )
    
    # Plot results
    plot_results(results_df, output_dir='results/cf_benchmark')
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print("\nKey Questions Answered:")
    print("1. Does CF transformation improve over simple averaging?")
    print("2. Does CF transformation improve over stacking?")
    print("3. What's the best ρ value (reconstruction vs. supervised)?")
    print("4. How does performance vary with class imbalance?")
    print("\nCheck results/cf_benchmark/ for detailed results and plots.")


if __name__ == '__main__':
    main()
