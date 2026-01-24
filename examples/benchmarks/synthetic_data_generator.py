"""
Synthetic Ensemble Data Generator
==================================

Flexible generation of synthetic probability matrices with controlled properties.

**Features**:
- Adjustable latent structure (known ground truth)
- Controlled noise levels
- Label-dependent biases
- Configurable classifier diversity
- Instance subgroup structure

**Use cases**:
- Benchmarking CF-Ensemble vs baselines
- Ablation studies (vary one property at a time)
- Understanding method behavior

**Phase**: 4.1 (Experimental Validation)
"""

import numpy as np
import argparse
from pathlib import Path
import json
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def generate_synthetic_ensemble(
    n_samples: int = 1000,
    n_classifiers: int = 10,
    n_labeled: int = 500,
    latent_dim: int = 5,
    noise_level: float = 0.1,
    label_bias: float = 0.2,
    diversity_level: str = 'medium',
    subgroup_structure: bool = True,
    n_subgroups: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic ensemble data with known latent structure.
    
    Parameters
    ----------
    n_samples : int
        Total number of instances
    n_classifiers : int
        Number of base classifiers
    n_labeled : int
        Number of labeled instances
    latent_dim : int
        Dimensionality of latent space
    noise_level : float
        Standard deviation of additive Gaussian noise
    label_bias : float
        How much to shift probabilities based on true label
        (larger = easier problem)
    diversity_level : str
        'low', 'medium', 'high' - controls classifier diversity
    subgroup_structure : bool
        Whether to add instance subgroups with varying difficulty
    n_subgroups : int
        Number of instance subgroups (if subgroup_structure=True)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    R : (n_classifiers, n_samples) array
        Probability matrix
    labels : (n_samples,) array
        Ground truth labels with NaN for unlabeled
    X_true : (latent_dim, n_classifiers) array
        True classifier factors
    Y_true : (latent_dim, n_samples) array
        True instance factors
    metadata : dict
        Additional information about the generated data
    """
    np.random.seed(seed)
    
    # 1. Generate true latent factors
    X_true = np.random.randn(latent_dim, n_classifiers)
    Y_true = np.random.randn(latent_dim, n_samples)
    
    # Normalize factors
    X_true = X_true / np.linalg.norm(X_true, axis=0, keepdims=True)
    Y_true = Y_true / np.linalg.norm(Y_true, axis=0, keepdims=True)
    
    # 2. Generate labels from instance factors
    # Use first principal component + noise
    label_logits = np.sum(Y_true[:2, :], axis=0) + np.random.randn(n_samples) * 0.3
    y_true = (label_logits > 0).astype(float)
    
    # 3. Generate clean probabilities from matrix factorization
    R_clean = sigmoid(X_true.T @ Y_true)  # (n_classifiers, n_samples)
    
    # 4. Add label-dependent bias (makes problem learnable)
    for i in range(n_samples):
        if y_true[i] == 1:
            # Shift probabilities up for positive class
            R_clean[:, i] = np.clip(R_clean[:, i] + label_bias, 0, 1)
        else:
            # Shift probabilities down for negative class
            R_clean[:, i] = np.clip(R_clean[:, i] - label_bias, 0, 1)
    
    # 5. Add classifier diversity
    diversity_map = {
        'low': (0.95, 0.02),    # (correlation, std of quality)
        'medium': (0.70, 0.10),
        'high': (0.50, 0.15)
    }
    target_corr, quality_std = diversity_map.get(diversity_level, diversity_map['medium'])
    
    # Generate diverse classifiers by adding per-classifier bias
    classifier_qualities = np.random.normal(0.75, quality_std, n_classifiers)
    classifier_qualities = np.clip(classifier_qualities, 0.55, 0.95)
    
    for u in range(n_classifiers):
        # Each classifier has a quality-dependent shift
        quality_shift = (classifier_qualities[u] - 0.75) * 0.3
        R_clean[u, :] += quality_shift
        R_clean[u, :] = np.clip(R_clean[u, :], 0, 1)
    
    # 6. Add subgroup structure
    if subgroup_structure:
        subgroup_assignment = np.random.randint(0, n_subgroups, n_samples)
        subgroup_difficulty = np.random.uniform(0.1, 0.5, n_subgroups)
        
        # Each classifier has varying performance across subgroups
        classifier_subgroup_affinity = np.random.uniform(0.7, 1.3, (n_classifiers, n_subgroups))
        
        for u in range(n_classifiers):
            for i in range(n_samples):
                subgroup = subgroup_assignment[i]
                difficulty = subgroup_difficulty[subgroup]
                affinity = classifier_subgroup_affinity[u, subgroup]
                
                # Adjust probability based on affinity and difficulty
                adjustment = (affinity - 1.0) * 0.2 - difficulty * 0.3
                R_clean[u, i] += adjustment
                R_clean[u, i] = np.clip(R_clean[u, i], 0, 1)
        
        metadata_subgroups = {
            'subgroup_assignment': subgroup_assignment.tolist(),
            'subgroup_difficulty': subgroup_difficulty.tolist(),
            'classifier_subgroup_affinity': classifier_subgroup_affinity.tolist()
        }
    else:
        subgroup_assignment = None
        metadata_subgroups = {}
    
    # 7. Add noise
    noise = np.random.randn(n_classifiers, n_samples) * noise_level
    R = np.clip(R_clean + noise, 0.01, 0.99)
    
    # 8. Create labeled/unlabeled split
    labels = y_true.copy()
    unlabeled_idx = np.random.choice(
        n_samples, size=n_samples - n_labeled, replace=False
    )
    labels[unlabeled_idx] = np.nan
    
    # 9. Compute actual statistics for metadata
    labeled_mask = ~np.isnan(labels)
    classifier_accuracies = []
    for u in range(n_classifiers):
        preds = (R[u, labeled_mask] > 0.5).astype(int)
        acc = np.mean(preds == y_true[labeled_mask])
        classifier_accuracies.append(acc)
    
    # Inter-classifier correlation
    classifier_corr = np.corrcoef(R)
    avg_correlation = (np.sum(classifier_corr) - n_classifiers) / (n_classifiers * (n_classifiers - 1))
    
    # Package metadata
    metadata = {
        'n_samples': n_samples,
        'n_classifiers': n_classifiers,
        'n_labeled': n_labeled,
        'latent_dim': latent_dim,
        'noise_level': noise_level,
        'label_bias': label_bias,
        'diversity_level': diversity_level,
        'subgroup_structure': subgroup_structure,
        'n_subgroups': n_subgroups if subgroup_structure else None,
        'seed': seed,
        'actual_avg_accuracy': float(np.mean(classifier_accuracies)),
        'actual_accuracy_std': float(np.std(classifier_accuracies)),
        'actual_avg_correlation': float(avg_correlation),
        'classifier_accuracies': [float(acc) for acc in classifier_accuracies],
        **metadata_subgroups
    }
    
    return R, labels, X_true, Y_true, metadata


def visualize_synthetic_data(
    R: np.ndarray,
    labels: np.ndarray,
    X_true: np.ndarray,
    Y_true: np.ndarray,
    metadata: Dict,
    output_path: Path
):
    """
    Create comprehensive visualization of synthetic data.
    
    Parameters
    ----------
    R : (n_classifiers, n_samples) array
        Probability matrix
    labels : (n_samples,) array
        Labels (with NaN for unlabeled)
    X_true, Y_true : arrays
        True latent factors
    metadata : dict
        Data statistics
    output_path : Path
        Where to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    labeled_mask = ~np.isnan(labels)
    y_true = labels[labeled_mask]
    
    # 1. Probability matrix heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(R, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax1.set_title(f'Probability Matrix ({R.shape[0]} classifiers × {R.shape[1]} instances)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Instances')
    ax1.set_ylabel('Classifiers')
    plt.colorbar(im, ax=ax1, label='Probability')
    
    # 2. Classifier accuracy distribution
    ax2 = fig.add_subplot(gs[1, 0])
    accuracies = metadata['classifier_accuracies']
    ax2.hist(accuracies, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Count')
    ax2.set_title('Classifier Quality Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Classifier correlation matrix
    ax3 = fig.add_subplot(gs[1, 1])
    corr = np.corrcoef(R)
    im = ax3.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
    ax3.set_title('Inter-Classifier Correlation', fontweight='bold')
    ax3.set_xlabel('Classifier')
    ax3.set_ylabel('Classifier')
    plt.colorbar(im, ax=ax3)
    
    # 4. Label distribution
    ax4 = fig.add_subplot(gs[1, 2])
    unique, counts = np.unique(labels[labeled_mask], return_counts=True)
    ax4.bar(['Negative', 'Positive'], counts, color=['salmon', 'lightgreen'], 
            edgecolor='black', alpha=0.7)
    ax4.set_ylabel('Count')
    ax4.set_title('Label Distribution (Labeled Only)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Per-instance prediction variance
    ax5 = fig.add_subplot(gs[2, 0])
    instance_std = np.std(R, axis=0)
    ax5.hist(instance_std, bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax5.axvline(np.mean(instance_std), color='red', linestyle='--',
                label=f'Mean: {np.mean(instance_std):.3f}')
    ax5.set_xlabel('Std Dev of Predictions')
    ax5.set_ylabel('Count')
    ax5.set_title('Instance-Level Disagreement', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Latent space (PCA of Y_true)
    ax6 = fig.add_subplot(gs[2, 1])
    if Y_true.shape[0] >= 2:
        ax6.scatter(Y_true[0, labeled_mask], Y_true[1, labeled_mask], 
                   c=y_true, cmap='RdYlBu', alpha=0.6, s=20)
        ax6.set_xlabel('Latent Dim 1')
        ax6.set_ylabel('Latent Dim 2')
        ax6.set_title('True Instance Latent Space', fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    # 7. Data statistics text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    stats_text = f"""
    DATA STATISTICS
    
    Problem Size:
      Classifiers: {metadata['n_classifiers']}
      Instances: {metadata['n_samples']}
      Labeled: {metadata['n_labeled']} ({metadata['n_labeled']/metadata['n_samples']*100:.1f}%)
      
    Generation Parameters:
      Latent dim: {metadata['latent_dim']}
      Noise level: {metadata['noise_level']:.3f}
      Label bias: {metadata['label_bias']:.3f}
      Diversity: {metadata['diversity_level']}
      
    Actual Properties:
      Avg accuracy: {metadata['actual_avg_accuracy']:.3f}
      Accuracy std: {metadata['actual_accuracy_std']:.3f}
      Avg correlation: {metadata['actual_avg_correlation']:.3f}
    """
    
    if metadata.get('subgroup_structure'):
        stats_text += f"\n  Subgroups: {metadata['n_subgroups']}"
    
    ax7.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax7.transAxes)
    
    plt.suptitle('Synthetic Ensemble Data Overview', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


def save_data(
    R: np.ndarray,
    labels: np.ndarray,
    X_true: np.ndarray,
    Y_true: np.ndarray,
    metadata: Dict,
    output_dir: Path
):
    """Save all data and metadata to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(output_dir / 'R.npy', R)
    np.save(output_dir / 'labels.npy', labels)
    np.save(output_dir / 'X_true.npy', X_true)
    np.save(output_dir / 'Y_true.npy', Y_true)
    
    # Save metadata as JSON
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Data saved to {output_dir}/")
    print(f"  - R.npy: {R.shape}")
    print(f"  - labels.npy: {labels.shape}")
    print(f"  - X_true.npy: {X_true.shape}")
    print(f"  - Y_true.npy: {Y_true.shape}")
    print(f"  - metadata.json")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic ensemble data')
    
    # Data size
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Total number of instances')
    parser.add_argument('--n-classifiers', type=int, default=10,
                       help='Number of base classifiers')
    parser.add_argument('--n-labeled', type=int, default=500,
                       help='Number of labeled instances')
    
    # Latent structure
    parser.add_argument('--latent-dim', type=int, default=5,
                       help='Latent dimensionality')
    
    # Noise and bias
    parser.add_argument('--noise-level', type=float, default=0.1,
                       help='Gaussian noise std dev')
    parser.add_argument('--label-bias', type=float, default=0.2,
                       help='Label-dependent probability shift')
    
    # Diversity
    parser.add_argument('--diversity', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='Classifier diversity level')
    
    # Subgroups
    parser.add_argument('--subgroups', action='store_true',
                       help='Add instance subgroup structure')
    parser.add_argument('--n-subgroups', type=int, default=4,
                       help='Number of subgroups (if --subgroups)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/benchmarks/data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Synthetic Ensemble Data Generator")
    print("="*60)
    print(f"Configuration:")
    print(f"  Problem size: {args.n_classifiers} classifiers × {args.n_samples} instances")
    print(f"  Labeled: {args.n_labeled} ({args.n_labeled/args.n_samples*100:.1f}%)")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Noise level: {args.noise_level}")
    print(f"  Label bias: {args.label_bias}")
    print(f"  Diversity: {args.diversity}")
    print(f"  Subgroups: {args.subgroups} (n={args.n_subgroups if args.subgroups else 'N/A'})")
    print(f"  Seed: {args.seed}")
    print("="*60)
    
    # Generate data
    R, labels, X_true, Y_true, metadata = generate_synthetic_ensemble(
        n_samples=args.n_samples,
        n_classifiers=args.n_classifiers,
        n_labeled=args.n_labeled,
        latent_dim=args.latent_dim,
        noise_level=args.noise_level,
        label_bias=args.label_bias,
        diversity_level=args.diversity,
        subgroup_structure=args.subgroups,
        n_subgroups=args.n_subgroups,
        seed=args.seed
    )
    
    print("\n✓ Data generated successfully!")
    print(f"  Average classifier accuracy: {metadata['actual_avg_accuracy']:.3f}")
    print(f"  Accuracy std: {metadata['actual_accuracy_std']:.3f}")
    print(f"  Average correlation: {metadata['actual_avg_correlation']:.3f}")
    
    # Save data
    output_dir = Path(args.output_dir)
    save_data(R, labels, X_true, Y_true, metadata, output_dir)
    
    # Visualize
    if args.visualize:
        print("\nGenerating visualization...")
        visualize_synthetic_data(
            R, labels, X_true, Y_true, metadata,
            output_dir / 'data_overview.png'
        )
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Review data_overview.png (if --visualize)")
    print("  2. Use this data with baseline_comparison.py")
    print("  3. Run ablation studies")


if __name__ == '__main__':
    main()
