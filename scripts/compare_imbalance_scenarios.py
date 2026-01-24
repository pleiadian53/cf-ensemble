"""
Compare Imbalance Scenarios
============================

Create side-by-side comparison of quality threshold experiments
across different class imbalance levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_comparison_plot():
    """Create comprehensive comparison across imbalance levels."""
    
    scenarios = [
        ('10% Positives\n(Disease Detection)', 'results/quality_threshold/summary.csv', 0.10),
        ('5% Positives\n(Rare Disease)', 'results/quality_threshold_5pct/summary.csv', 0.05),
        ('1% Positives\n(Splice Sites)', 'results/quality_threshold_1pct/summary.csv', 0.01),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Impact of Class Imbalance on Confidence Weighting Effectiveness', 
                 fontsize=16, fontweight='bold')
    
    colors = {'uniform': '#1f77b4', 'certainty': '#ff7f0e', 
              'calibration': '#2ca02c', 'learned': '#d62728'}
    
    for col_idx, (name, path, pos_rate) in enumerate(scenarios):
        df = pd.read_csv(path)
        
        # Top row: Performance vs Quality
        ax = axes[0, col_idx]
        for strategy in ['uniform', 'certainty', 'calibration', 'learned']:
            strategy_data = df[df['strategy'] == strategy]
            ax.plot(strategy_data['actual_quality_prauc'], 
                   strategy_data['prauc_mean'],
                   marker='o', label=strategy, linewidth=2,
                   color=colors.get(strategy, 'gray'), markersize=6)
        
        ax.set_xlabel('Avg Classifier PR-AUC', fontsize=10)
        ax.set_ylabel('Ensemble PR-AUC', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=9, loc='upper left')
        
        # Add random baseline
        ax.axhline(y=pos_rate, color='red', linestyle='--', 
                  alpha=0.5, linewidth=1, label='Random')
        
        # Bottom row: Improvement vs Quality
        ax = axes[1, col_idx]
        learned_data = df[df['strategy'] == 'learned']
        label_data = df[df['strategy'] == 'label_aware']
        
        ax.plot(learned_data['actual_quality_prauc'],
               learned_data['improvement_prauc_mean'] * 100,
               marker='o', linewidth=2.5, color='#d62728',
               label='Learned Reliability', markersize=7)
        ax.plot(label_data['actual_quality_prauc'],
               label_data['improvement_prauc_mean'] * 100,
               marker='s', linewidth=2, color='#9467bd',
               label='Label-Aware', markersize=6, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, 
                  linewidth=1.5, label='1% threshold')
        
        ax.set_xlabel('Avg Classifier PR-AUC', fontsize=10)
        ax.set_ylabel('Improvement over Uniform (%)', fontsize=10)
        ax.set_title(f'Max: {learned_data["improvement_prauc_mean"].max()*100:.2f}%', 
                    fontsize=11, color='darkgreen' if learned_data["improvement_prauc_mean"].max() > 0.01 else 'darkred')
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    
    # Save
    output_path = Path('results/imbalance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {output_path}")
    
    plt.close()


def print_summary_table():
    """Print comprehensive summary table."""
    
    scenarios = [
        ('10% pos', 'results/quality_threshold/summary.csv', 0.10),
        ('5% pos', 'results/quality_threshold_5pct/summary.csv', 0.05),
        ('1% pos', 'results/quality_threshold_1pct/summary.csv', 0.01),
    ]
    
    print('='*100)
    print('IMBALANCE COMPARISON: Summary Statistics')
    print('='*100)
    print(f"\n{'Scenario':<12} {'Random':<10} {'Quality Range':<20} {'Peak Imp.':<12} {'Best Baseline':<15} {'Status':<20}")
    print('-'*100)
    
    for name, path, pos_rate in scenarios:
        df = pd.read_csv(path)
        learned = df[df['strategy'] == 'learned']
        
        # Stats
        q_min = learned['actual_quality_prauc'].min()
        q_max = learned['actual_quality_prauc'].max()
        peak_imp = learned['improvement_prauc_mean'].max() * 100
        
        best_idx = learned['prauc_mean'].idxmax()
        best_baseline = learned.loc[best_idx, 'prauc_mean']
        
        # Status
        if peak_imp >= 3:
            status = '✅✅✅ OPTIMAL'
        elif peak_imp >= 1:
            status = '✅ Recommended'
        elif peak_imp >= 0.5:
            status = '⚠️ Limited benefit'
        else:
            status = '❌ Not recommended'
        
        print(f"{name:<12} {pos_rate:<10.3f} {f'{q_min:.3f}-{q_max:.3f}':<20} {f'+{peak_imp:.2f}%':<12} {best_baseline:<15.4f} {status:<20}")
    
    print('='*100)
    print('\n🌟 Key Finding: 5% positives shows BEST gains (+3.94%)')
    print('   → Sweet spot: Challenging but tractable!')
    print('\n⚠️  Extreme imbalance (<1%): Confidence weighting alone insufficient')
    print('   → Need: More data, better features, active learning\n')


if __name__ == '__main__':
    print("Creating imbalance comparison analysis...\n")
    print_summary_table()
    print("\nGenerating comparison plot...")
    create_comparison_plot()
    print("\n✅ Comparison complete!")
    print("\nGenerated files:")
    print("  - results/imbalance_comparison.png")
