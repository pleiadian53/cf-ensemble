"""
Quality Threshold Experiment
============================

Systematically vary base classifier quality to empirically determine:
1. Minimum viable quality threshold
2. Optimal quality range (sweet spot)
3. Diminishing returns threshold
4. Improvement vs quality curve

**Key Features:**
- **Class Imbalanced Data**: 10% positives (minority class), 90% negatives
  - Mimics realistic biomedical scenarios: disease detection, drug response, rare events
- **Primary Metric**: PR-AUC (Precision-Recall AUC) - focuses on minority class performance
- **Reference Metric**: ROC-AUC - for comparison with literature
- **Additional Metrics**: F1-Score - for operational thresholds

This experiment validates (or refutes) hypothesized thresholds for imbalanced data:
- When does confidence weighting start helping?
- What's the optimal quality range (sweet spot)?
- When do we hit diminishing returns?

Note: Thresholds will differ from balanced data. With 10% positives:
- PR-AUC is more informative than ROC-AUC
- Baseline performance will be lower (finding rare positives is hard!)
- Small PR-AUC improvements are meaningful
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import pandas as pd

from cfensemble.data import (
    EnsembleData,
    get_confidence_strategy,
    generate_imbalanced_ensemble_data
)
from cfensemble.models import ReliabilityWeightModel
from cfensemble.optimization import CFEnsembleTrainer
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    auc as sk_auc,
    f1_score
)


# Wrapper function to maintain compatibility with this script
# Uses the reusable synthetic data generation module
def generate_controlled_quality_data(
    avg_quality: float,
    diversity: str = 'high',
    n_classifiers: int = 15,
    n_instances: int = 300,
    n_labeled: int = 150,
    positive_rate: float = 0.10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic data with controlled average quality.
    
    This is a wrapper around cfensemble.data.generate_imbalanced_ensemble_data().
    See that function for full documentation.
    
    Parameters
    ----------
    avg_quality : float
        Target average classifier quality
    diversity : str
        'low', 'medium', or 'high'
    n_classifiers : int
        Number of base classifiers
    n_instances : int
        Total instances
    n_labeled : int
        Number of labeled instances
    positive_rate : float, default=0.10
        Fraction of positive (minority class) instances
    random_state : int
        Random seed
        
    Returns
    -------
    R, labels, labeled_idx, y_true : tuple
        See cfensemble.data.generate_imbalanced_ensemble_data() for details
    """
    return generate_imbalanced_ensemble_data(
        n_classifiers=n_classifiers,
        n_instances=n_instances,
        n_labeled=n_labeled,
        positive_rate=positive_rate,
        target_quality=avg_quality,
        diversity=diversity,
        random_state=random_state
    )


def evaluate_all_strategies(
    R: np.ndarray,
    labels: np.ndarray,
    labeled_idx: np.ndarray,
    y_true: np.ndarray,
    rho: float = 0.5,
    max_iter: int = 50,
    primary_metric: str = 'prauc'
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate all confidence strategies.
    
    Parameters
    ----------
    primary_metric : str, default='prauc'
        Primary evaluation metric: 'prauc' (recommended for imbalanced) or 'roc_auc'
    
    Returns
    -------
    results : dict
        Strategy name -> {'prauc': score, 'roc_auc': score, 'f1': score}
    """
    m, n = R.shape
    results = {}
    
    # Helper function to compute all metrics
    def compute_metrics(y_true, y_pred_proba):
        metrics = {}
        # PR-AUC (primary for imbalanced data)
        try:
            metrics['prauc'] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            metrics['prauc'] = 0.0
        # ROC-AUC (reference)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5
        # F1-Score
        try:
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            metrics['f1'] = f1_score(y_true, y_pred_binary)
        except ValueError:
            metrics['f1'] = 0.0
        return metrics
    
    # 1. Uniform baseline
    strategy = get_confidence_strategy('uniform')
    C = strategy.compute(R, labels)
    ensemble_data = EnsembleData(R, labels, C=C)
    trainer = CFEnsembleTrainer(n_classifiers=m, rho=rho, max_iter=max_iter, verbose=0)
    trainer.fit(ensemble_data)
    y_pred = trainer.predict()
    results['uniform'] = compute_metrics(y_true, y_pred)
    
    # 2. Fixed strategies
    for strategy_name in ['certainty', 'label_aware', 'calibration', 'adaptive']:
        strategy = get_confidence_strategy(strategy_name)
        C = strategy.compute(R, labels)
        ensemble_data = EnsembleData(R, labels, C=C)
        trainer = CFEnsembleTrainer(n_classifiers=m, rho=rho, max_iter=max_iter, verbose=0)
        trainer.fit(ensemble_data)
        y_pred = trainer.predict()
        results[strategy_name] = compute_metrics(y_true, y_pred)
    
    # 3. Learned reliability
    # Convert labeled_idx to boolean mask
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    # Compute classifier statistics (format: stat_name -> array of shape (m,))
    # Focus on metrics robust to class imbalance
    roc_auc = np.zeros(m)
    pr_auc = np.zeros(m)
    ap_score = np.zeros(m)
    f1 = np.zeros(m)
    avg_confidence = np.zeros(m)
    
    for u in range(m):
        y_true_labeled = labels[labeled_mask]
        y_pred_proba = R[u, labeled_mask]
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # ROC-AUC (baseline reference, less sensitive to imbalance)
        try:
            roc_auc[u] = roc_auc_score(y_true_labeled, y_pred_proba)
        except ValueError:
            roc_auc[u] = 0.5  # Only one class present
        
        # PR-AUC (Precision-Recall AUC - better for imbalanced data)
        try:
            precision, recall, _ = precision_recall_curve(y_true_labeled, y_pred_proba)
            pr_auc[u] = sk_auc(recall, precision)
        except ValueError:
            pr_auc[u] = 0.0
        
        # Average Precision (AP) - another good metric for imbalanced data
        try:
            ap_score[u] = average_precision_score(y_true_labeled, y_pred_proba)
        except ValueError:
            ap_score[u] = 0.0
        
        # F1-score
        try:
            f1[u] = f1_score(y_true_labeled, y_pred_binary)
        except ValueError:
            f1[u] = 0.0
        
        # Average confidence
        avg_confidence[u] = np.mean(np.abs(y_pred_proba - 0.5))
    
    classifier_stats = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': ap_score,
        'f1': f1,
        'avg_confidence': avg_confidence
    }
    
    rel_model = ReliabilityWeightModel(model_type='gbm', n_estimators=30)
    rel_model.fit(R, labels, labeled_mask, classifier_stats)
    W = rel_model.predict(R, classifier_stats)
    
    ensemble_data = EnsembleData(R, labels, C=W)
    trainer = CFEnsembleTrainer(n_classifiers=m, rho=rho, max_iter=max_iter, verbose=0)
    trainer.fit(ensemble_data)
    y_pred = trainer.predict()
    results['learned'] = compute_metrics(y_true, y_pred)
    
    # Add classifier statistics summary
    results['_classifier_stats'] = {
        'avg_prauc': np.mean(pr_auc),
        'avg_roc_auc': np.mean(roc_auc),
        'avg_f1': np.mean(f1)
    }
    
    return results


def run_quality_sweep(
    quality_levels: List[float],
    diversity: str = 'high',
    n_trials: int = 5,
    positive_rate: float = 0.10,
    output_dir: Path = Path("results/quality_threshold")
) -> pd.DataFrame:
    """
    Run full quality sweep experiment.
    
    Parameters
    ----------
    quality_levels : list
        Average quality levels to test (e.g., [0.50, 0.55, ..., 0.95])
    diversity : str
        Classifier diversity level
    n_trials : int
        Number of random trials per quality level
    positive_rate : float
        Fraction of positive (minority class) instances
    output_dir : Path
        Where to save results
        
    Returns
    -------
    results_df : DataFrame
        Full results table
    """
    print("="*60)
    print("Quality Threshold Experiment (Imbalanced Data)")
    print("="*60)
    print(f"Class distribution: {(1-positive_rate)*100:.0f}% negatives, {positive_rate*100:.0f}% positives (minority)")
    print(f"Quality levels: {quality_levels}")
    print(f"Diversity: {diversity}")
    print(f"Trials per level: {n_trials}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for quality in quality_levels:
        print(f"\n{'='*60}")
        print(f"Testing Quality: {quality:.2f} ({quality*100:.0f}%)")
        print(f"{'='*60}")
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)
            
            # Generate data
            R, labels, labeled_idx, y_true = generate_controlled_quality_data(
                avg_quality=quality,
                diversity=diversity,
                positive_rate=positive_rate,
                random_state=42 + trial
            )
            
            # Verify actual quality (use PR-AUC as primary metric for imbalanced data)
            m, n = R.shape
            
            # Convert labeled_idx to boolean mask for indexing
            labeled_mask_verify = np.zeros(n, dtype=bool)
            labeled_mask_verify[labeled_idx] = True
            
            actual_prauc = []
            actual_roc_aucs = []
            for u in range(m):
                y_true_labeled = labels[labeled_mask_verify]
                y_pred_proba = R[u, labeled_mask_verify]
                
                # PR-AUC (primary for imbalanced data)
                try:
                    prauc_u = average_precision_score(y_true_labeled, y_pred_proba)
                except ValueError:
                    prauc_u = 0.0  # Only one class
                actual_prauc.append(prauc_u)
                
                # ROC-AUC (reference)
                try:
                    roc_auc_u = roc_auc_score(y_true_labeled, y_pred_proba)
                except ValueError:
                    roc_auc_u = 0.5  # Only one class
                actual_roc_aucs.append(roc_auc_u)
                
            actual_avg_quality_prauc = np.mean(actual_prauc)
            actual_avg_quality_roc = np.mean(actual_roc_aucs)
            
            # Evaluate strategies
            strategy_results = evaluate_all_strategies(R, labels, labeled_idx, y_true)
            
            # Store results
            for strategy, metrics in strategy_results.items():
                if strategy == '_classifier_stats':  # Skip summary stats
                    continue
                    
                all_results.append({
                    'target_quality': quality,
                    'actual_quality_prauc': actual_avg_quality_prauc,
                    'actual_quality_roc': actual_avg_quality_roc,
                    'diversity': diversity,
                    'trial': trial,
                    'strategy': strategy,
                    'prauc': metrics['prauc'],
                    'roc_auc': metrics['roc_auc'],
                    'f1': metrics['f1'],
                    'improvement_prauc': metrics['prauc'] - strategy_results['uniform']['prauc'],
                    'improvement_roc': metrics['roc_auc'] - strategy_results['uniform']['roc_auc']
                })
            
            print(f"Done. Avg quality (PR-AUC): {actual_avg_quality_prauc:.3f} | "
                  f"Baseline: {strategy_results['uniform']['prauc']:.4f}, "
                  f"Learned: {strategy_results['learned']['prauc']:.4f} "
                  f"(+{(strategy_results['learned']['prauc'] - strategy_results['uniform']['prauc'])*100:.1f}%)")
        
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv(output_dir / 'raw_results.csv', index=False)
    print(f"\n✓ Raw results saved to {output_dir / 'raw_results.csv'}")
    
    return results_df


def analyze_and_plot(results_df: pd.DataFrame, output_dir: Path):
    """
    Analyze results and generate plots.
    """
    print("\n" + "="*60)
    print("Analyzing Results")
    print("="*60)
    
    # Aggregate by quality level
    summary = results_df.groupby(['target_quality', 'strategy']).agg({
        'actual_quality_prauc': 'mean',
        'actual_quality_roc': 'mean',
        'prauc': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'improvement_prauc': ['mean', 'std'],
        'improvement_roc': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['target_quality', 'strategy', 
                       'actual_quality_prauc', 'actual_quality_roc',
                       'prauc_mean', 'prauc_std',
                       'roc_auc_mean', 'roc_auc_std',
                       'f1_mean', 'f1_std',
                       'improvement_prauc_mean', 'improvement_prauc_std',
                       'improvement_roc_mean', 'improvement_roc_std']
    
    # Save summary
    summary.to_csv(output_dir / 'summary.csv', index=False)
    print(f"✓ Summary saved to {output_dir / 'summary.csv'}")
    
    # Print key findings
    print("\n" + "-"*60)
    print("KEY FINDINGS (Primary Metric: PR-AUC for Imbalanced Data)")
    print("-"*60)
    
    # Class imbalance info
    pos_rate = np.mean(results_df[results_df['trial'] == 0]['actual_quality_prauc'] < 
                       results_df[results_df['trial'] == 0]['actual_quality_roc'])
    print(f"📊 Data: Class imbalanced (10% positives, 90% negatives) - realistic biomedical scenario")
    print(f"   Examples: disease detection, drug response, rare events")
    print(f"   PR-AUC is primary metric (focuses on minority class), ROC-AUC for reference\n")
    
    # Find minimum viable quality (where learned > uniform by at least 1%)
    learned_results = summary[summary['strategy'] == 'learned']
    viable = learned_results[learned_results['improvement_prauc_mean'] >= 0.01]
    if len(viable) > 0:
        min_viable = viable['actual_quality_prauc'].min()
        print(f"✓ Minimum viable quality: {min_viable:.3f} PR-AUC ({min_viable*100:.1f}%)")
    else:
        print("✗ No quality level achieved >1% improvement")
    
    # Find peak improvement
    peak_idx = learned_results['improvement_prauc_mean'].idxmax()
    peak_quality = learned_results.loc[peak_idx, 'actual_quality_prauc']
    peak_improvement = learned_results.loc[peak_idx, 'improvement_prauc_mean']
    print(f"✓ Peak improvement: {peak_improvement*100:.2f}% at quality {peak_quality:.3f} PR-AUC ({peak_quality*100:.1f}%)")
    
    # Find diminishing returns (where improvement < 1%)
    diminishing = learned_results[learned_results['improvement_prauc_mean'] < 0.01]
    if len(diminishing) > 0:
        diminishing_threshold = diminishing['actual_quality_prauc'].min()
        print(f"✓ Diminishing returns below: {diminishing_threshold:.3f} PR-AUC ({diminishing_threshold*100:.1f}%)")
        print(f"   (Confidence weighting provides < 1% improvement when quality < {diminishing_threshold:.3f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quality Threshold Experiment Results (Imbalanced Data: 10% Positives, 90% Negatives)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: PR-AUC vs Quality (all strategies)
    ax = axes[0, 0]
    for strategy in ['uniform', 'certainty', 'calibration', 'learned']:
        strategy_data = summary[summary['strategy'] == strategy]
        ax.plot(strategy_data['actual_quality_prauc'], strategy_data['prauc_mean'], 
                marker='o', label=strategy, linewidth=2)
        ax.fill_between(strategy_data['actual_quality_prauc'],
                        strategy_data['prauc_mean'] - strategy_data['prauc_std'],
                        strategy_data['prauc_mean'] + strategy_data['prauc_std'],
                        alpha=0.2)
    ax.set_xlabel('Average Classifier PR-AUC', fontsize=12)
    ax.set_ylabel('Ensemble PR-AUC', fontsize=12)
    ax.set_title('Performance vs Quality (Primary: PR-AUC)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Improvement vs Quality (learned only)
    ax = axes[0, 1]
    learned_data = summary[summary['strategy'] == 'learned']
    ax.plot(learned_data['actual_quality_prauc'], learned_data['improvement_prauc_mean'] * 100,
            marker='o', linewidth=2, color='#2E7D32', label='Learned Reliability')
    ax.fill_between(learned_data['actual_quality_prauc'],
                    (learned_data['improvement_prauc_mean'] - learned_data['improvement_prauc_std']) * 100,
                    (learned_data['improvement_prauc_mean'] + learned_data['improvement_prauc_std']) * 100,
                    alpha=0.3, color='#2E7D32')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1% threshold')
    ax.set_xlabel('Average Classifier PR-AUC', fontsize=12)
    ax.set_ylabel('PR-AUC Improvement over Uniform (%)', fontsize=12)
    ax.set_title('Learned Reliability Improvement', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Strategy comparison at different quality levels
    ax = axes[1, 0]
    quality_bins = [0.25, 0.35, 0.45, 0.55]  # Adjusted for PR-AUC scale
    bin_labels = ['25-30%', '35-40%', '45-50%', '55-60%']
    strategies = ['uniform', 'certainty', 'calibration', 'learned']
    
    for i, (q_low, q_label) in enumerate(zip(quality_bins, bin_labels)):
        q_high = q_low + 0.10
        bin_data = summary[(summary['actual_quality_prauc'] >= q_low) & 
                          (summary['actual_quality_prauc'] < q_high)]
        
        if len(bin_data) > 0:
            improvements = [bin_data[bin_data['strategy'] == s]['improvement_prauc_mean'].mean() * 100
                           for s in strategies]
            x_pos = np.arange(len(strategies)) + i * 0.2
            ax.bar(x_pos, improvements, width=0.18, label=q_label)
    
    ax.set_xticks(np.arange(len(strategies)) + 0.3)
    ax.set_xticklabels(strategies, rotation=15)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Strategy Performance by Quality Range', fontsize=13, fontweight='bold')
    ax.legend(title='Quality Range')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Heatmap of improvement (PR-AUC)
    ax = axes[1, 1]
    pivot = summary[summary['strategy'].isin(['certainty', 'calibration', 'learned'])].pivot(
        index='strategy', columns='target_quality', values='improvement_prauc_mean'
    )
    im = ax.imshow(pivot.values * 100, aspect='auto', cmap='RdYlGn', 
                   vmin=-2, vmax=max(5, pivot.values.max() * 100))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{q:.0%}' for q in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Target Quality', fontsize=12)
    ax.set_title('Improvement Heatmap (%)', fontsize=13, fontweight='bold')
    
    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j] * 100
            color = 'white' if abs(value) > 2 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Improvement (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {output_dir / 'quality_threshold_analysis.png'}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Quality threshold experiment for imbalanced data')
    parser.add_argument('--diversity', type=str, default='high',
                       choices=['low', 'medium', 'high'],
                       help='Classifier diversity level')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per quality level')
    parser.add_argument('--positive-rate', type=float, default=0.10,
                       help='Fraction of positive (minority class) instances. '
                            'Examples: 0.10 (10%%, disease detection), 0.05 (5%%, rare disease), '
                            '0.01 (1%%, splice sites)')
    parser.add_argument('--output-dir', type=str, default='results/quality_threshold',
                       help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Quality levels to test
    quality_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    # Run experiment
    results_df = run_quality_sweep(
        quality_levels=quality_levels,
        diversity=args.diversity,
        n_trials=args.trials,
        positive_rate=args.positive_rate,
        output_dir=output_dir
    )
    
    # Analyze and plot
    analyze_and_plot(results_df, output_dir)
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  - raw_results.csv (full data)")
    print(f"  - summary.csv (aggregated)")
    print(f"  - quality_threshold_analysis.png (plots)")


if __name__ == '__main__':
    main()
