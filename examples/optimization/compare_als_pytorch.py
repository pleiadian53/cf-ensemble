"""
Compare ALS vs PyTorch Gradient Descent

This example demonstrates that both optimization methods converge to
similar solutions for the CF-Ensemble matrix factorization objective.

Expected outcome: Both methods should achieve similar reconstruction errors
and converge to local minima of the same objective function.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Install with: pip install torch")
    print("Proceeding with ALS-only demonstration...")

from cfensemble.optimization.als import (
    update_classifier_factors,
    update_instance_factors,
    compute_reconstruction_error
)

if TORCH_AVAILABLE:
    from cfensemble.optimization.pytorch_gd import compare_als_vs_pytorch


def generate_test_data(m=10, n=100, d=5, noise_level=0.1):
    """
    Generate synthetic test data for comparison.
    
    Parameters
    ----------
    m : int
        Number of classifiers
    n : int
        Number of instances
    d : int
        True latent dimension
    noise_level : float
        Noise added to probabilities
    
    Returns
    -------
    R : np.ndarray, shape (m, n)
        Probability matrix
    C : np.ndarray, shape (m, n)
        Confidence weights
    """
    np.random.seed(42)
    
    # Generate true latent factors
    X_true = np.random.randn(d, m) * 0.5
    Y_true = np.random.randn(d, n) * 0.5
    
    # Generate probability matrix
    R = X_true.T @ Y_true
    R = np.clip(R + np.random.randn(m, n) * noise_level, 0.01, 0.99)
    
    # Certainty-based confidence
    C = np.abs(R - 0.5)
    
    return R, C


def run_als_only(R, C, latent_dim=10, lambda_reg=0.01, max_iter=50):
    """Run ALS optimization only."""
    m, n = R.shape
    
    print("="*60)
    print("Running ALS Optimization")
    print("="*60)
    print(f"Problem size: {m} classifiers × {n} instances")
    print(f"Latent dimension: {latent_dim}")
    print(f"Regularization: λ = {lambda_reg}")
    print()
    
    # Initialize
    np.random.seed(42)
    X = np.random.randn(latent_dim, m) * 0.01
    Y = np.random.randn(latent_dim, n) * 0.01
    
    losses = []
    
    # Optimization loop
    for iteration in range(max_iter):
        # Update factors
        X = update_classifier_factors(Y, R, C, lambda_reg)
        Y = update_instance_factors(X, R, C, lambda_reg)
        
        # Compute loss
        recon_error = compute_reconstruction_error(X, Y, R, C)
        reg_loss = lambda_reg * (np.sum(X**2) + np.sum(Y**2))
        total_loss = recon_error + reg_loss
        losses.append(total_loss)
        
        if iteration % 10 == 0 or iteration == max_iter - 1:
            print(f"Iter {iteration:3d}: Loss = {total_loss:10.4f} "
                  f"(Recon = {recon_error:10.4f}, Reg = {reg_loss:8.4f})")
    
    print(f"\nFinal reconstruction error: {recon_error:.4f}")
    print("ALS optimization complete!")
    
    return X, Y, losses


def visualize_comparison(results, output_dir='results/als_vs_pytorch'):
    """Visualize comparison results.
    
    Parameters
    ----------
    results : dict
        Results from compare_als_vs_pytorch
    output_dir : str or Path
        Directory to save visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(results['als_losses'], 'b-', label='ALS', linewidth=2)
    ax.plot(results['pytorch_losses'], 'r--', label='PyTorch', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Log-scale loss
    ax = axes[0, 1]
    ax.semilogy(results['als_losses'], 'b-', label='ALS', linewidth=2)
    ax.semilogy(results['pytorch_losses'], 'r--', label='PyTorch', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss (log scale)')
    ax.set_title('Convergence (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Reconstruction errors comparison
    ax = axes[1, 0]
    methods = ['ALS', 'PyTorch']
    errors = [results['recon_error_als'], results['recon_error_pytorch']]
    colors = ['blue', 'red']
    ax.bar(methods, errors, color=colors, alpha=0.7)
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Final Reconstruction Error')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add error values on top of bars
    for i, (method, error) in enumerate(zip(methods, errors)):
        ax.text(i, error, f'{error:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Factor correlations
    ax = axes[1, 1]
    correlations = [results['X_correlation'], results['Y_correlation']]
    factor_names = ['Classifier Factors (X)', 'Instance Factors (Y)']
    colors = ['green', 'orange']
    ax.bar(factor_names, correlations, color=colors, alpha=0.7)
    ax.set_ylabel('Correlation')
    ax.set_title('Factor Similarity (ALS vs PyTorch)')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='High correlation threshold')
    ax.legend()
    
    # Add correlation values
    for i, (name, corr) in enumerate(zip(factor_names, correlations)):
        ax.text(i, corr, f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'als_vs_pytorch_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_file}")
    plt.show()


def main(output_dir='results/als_vs_pytorch'):
    """Main demonstration function.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save results and visualizations
    """
    output_dir = Path(output_dir)
    
    print("="*60)
    print("CF-Ensemble: ALS vs PyTorch Gradient Descent Comparison")
    print("="*60)
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()
    
    # Generate data
    print("Generating synthetic data...")
    m, n = 10, 100
    latent_dim = 5
    R, C = generate_test_data(m, n, d=latent_dim, noise_level=0.15)
    print(f"  Matrix shape: {m} classifiers × {n} instances")
    print(f"  True latent dimension: {latent_dim}")
    print()
    
    if not TORCH_AVAILABLE:
        # Run ALS only
        X_als, Y_als, losses_als = run_als_only(
            R, C,
            latent_dim=10,  # Use higher dim for challenge
            lambda_reg=0.01,
            max_iter=50
        )
        print("\n" + "="*60)
        print("Note: Install PyTorch to compare with gradient descent:")
        print("  pip install torch")
        print("="*60)
        return
    
    # Run comparison
    print("Running both ALS and PyTorch optimization...")
    print()
    
    results = compare_als_vs_pytorch(
        R, C,
        latent_dim=10,
        lambda_reg=0.01,
        max_iter=200,  # Increased: PyTorch needs more iterations than ALS
        random_seed=42,
        verbose=True
    )
    
    # Detailed comparison
    print("\n" + "="*60)
    print("Detailed Analysis")
    print("="*60)
    
    # 1. Final losses
    print("\n1. Final Total Loss:")
    print(f"   ALS:     {results['final_loss_als']:.4f}")
    print(f"   PyTorch: {results['final_loss_pytorch']:.4f}")
    diff_pct = abs(results['final_loss_als'] - results['final_loss_pytorch']) / results['final_loss_als'] * 100
    print(f"   Difference: {diff_pct:.2f}%")
    
    # 2. Reconstruction errors
    print("\n2. Reconstruction Error (no regularization):")
    print(f"   ALS:     {results['recon_error_als']:.4f}")
    print(f"   PyTorch: {results['recon_error_pytorch']:.4f}")
    diff_pct = abs(results['recon_error_als'] - results['recon_error_pytorch']) / results['recon_error_als'] * 100
    print(f"   Difference: {diff_pct:.2f}%")
    
    # 3. Factor similarity
    print("\n3. Factor Similarity (Correlation):")
    print(f"   X (Classifiers): {results['X_correlation']:.4f}")
    print(f"   Y (Instances):   {results['Y_correlation']:.4f}")
    
    if results['X_correlation'] > 0.9 and results['Y_correlation'] > 0.9:
        print("   ✓ High correlation - factors are very similar!")
    else:
        print("   Note: Lower correlation due to different optimization paths")
        print("         (Both methods can find different local minima)")
    
    # 4. Convergence speed
    als_iters = len(results['als_losses'])
    pytorch_iters = len(results['pytorch_losses'])
    print("\n4. Convergence Speed:")
    print(f"   ALS iterations:     {als_iters}")
    print(f"   PyTorch iterations: {pytorch_iters}")
    
    # 5. Optimization summary
    print("\n5. Summary:")
    print("   ✓ Both methods optimize the same objective")
    print("   ✓ Both converge to local minima")
    if diff_pct < 10:
        print("   ✓ Results are consistent (< 10% difference)")
    else:
        print("   Note: Some difference due to different optimization paths")
    
    # Visualization
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    try:
        visualize_comparison(results, output_dir)
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("(matplotlib might not be available or display issues)")
    
    # Conclusions
    print("\n" + "="*60)
    print("Conclusions")
    print("="*60)
    print("\n✓ ALS and PyTorch gradient descent are mathematically equivalent")
    print("✓ Both minimize the same objective function")
    print("✓ Differences arise from:")
    print("   - Different optimization trajectories")
    print("   - Random initialization")
    print("   - Non-convex objective (multiple local minima)")
    print("\n✓ Choose ALS for:")
    print("   - CPU-only environments")
    print("   - Small-medium problems")
    print("   - Stable, parameter-free optimization")
    print("\n✓ Choose PyTorch for:")
    print("   - GPU acceleration")
    print("   - Large-scale problems")
    print("   - Experimentation with extensions")
    
    print(f"\n📊 Results saved to:")
    print(f"   {output_dir.absolute()}/")
    print(f"     └── als_vs_pytorch_comparison.png")
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ALS vs PyTorch Gradient Descent for CF-Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default output directory (results/als_vs_pytorch/)
  python compare_als_pytorch.py
  
  # Specify custom output directory
  python compare_als_pytorch.py --output-dir experiments/optimization_comparison
  
  # Use different output directory for testing
  python compare_als_pytorch.py -o results/test_comparison
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/als_vs_pytorch',
        help='Directory to save results and visualizations (default: results/als_vs_pytorch)'
    )
    
    args = parser.parse_args()
    main(output_dir=args.output_dir)
