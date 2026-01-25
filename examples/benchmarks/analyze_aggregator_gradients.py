"""
Mathematical Analysis: Why Do Aggregator Weights Collapse?

Traces gradients step-by-step to find the root cause.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import generate_imbalanced_ensemble_data
from cfensemble.ensemble.aggregators import WeightedAggregator, sigmoid


def analyze_gradient_behavior():
    """Analyze what happens to gradients with imbalanced data."""
    print("="*70)
    print("GRADIENT ANALYSIS: Why Weights Collapse")
    print("="*70)
    
    # Simple setup: 5 classifiers, 100 instances, 10% positive
    np.random.seed(42)
    m, n = 5, 100
    
    # Generate good R_hat (excellent classifiers)
    y_true = (np.random.rand(n) < 0.10).astype(float)
    
    # Simulate good predictions: high for positives, low for negatives
    R_hat = np.zeros((m, n))
    for u in range(m):
        for i in range(n):
            if y_true[i] == 1:
                R_hat[u, i] = np.random.uniform(0.6, 0.9)  # Good: predict high
            else:
                R_hat[u, i] = np.random.uniform(0.1, 0.4)  # Good: predict low
    
    print(f"\nData Setup:")
    print(f"  Classifiers: {m}")
    print(f"  Instances: {n}")
    print(f"  Positive rate: {np.mean(y_true):.1%}")
    print(f"  R_hat range: [{R_hat.min():.2f}, {R_hat.max():.2f}]")
    
    # Initialize aggregator
    agg = WeightedAggregator(m, init_uniform=True)
    print(f"\nInitial weights: {agg.w}")
    print(f"Initial bias: {agg.b:.3f}")
    
    # Simulate training iterations
    print("\n" + "="*70)
    print("GRADIENT EVOLUTION")
    print("="*70)
    
    lr = 0.1
    
    for iteration in range(10):
        # Current predictions
        y_pred = agg.predict(R_hat)
        
        # Compute gradients (same as in aggregator.update)
        residual = y_pred - y_true
        grad_w = (R_hat @ residual) / len(residual)
        grad_b = np.mean(residual)
        
        # Analyze residuals
        residual_pos = residual[y_true == 1]
        residual_neg = residual[y_true == 0]
        
        print(f"\nIteration {iteration}:")
        print(f"  Weights: {agg.w.round(3)}")
        print(f"  Weight sum: {np.sum(agg.w):.3f}, std: {np.std(agg.w):.3f}")
        print(f"  Bias: {agg.b:.3f}")
        
        print(f"  Predictions: mean={np.mean(y_pred):.3f}, std={np.std(y_pred):.3f}")
        print(f"    On positives (n={len(residual_pos)}): {np.mean(y_pred[y_true==1]):.3f}")
        print(f"    On negatives (n={len(residual_neg)}): {np.mean(y_pred[y_true==0]):.3f}")
        
        print(f"  Residuals: mean={np.mean(residual):.4f}")
        print(f"    On positives: {np.mean(residual_pos):.4f}")
        print(f"    On negatives: {np.mean(residual_neg):.4f}")
        
        print(f"  Gradients:")
        print(f"    grad_w: {grad_w.round(4)}")
        print(f"    grad_w mean: {np.mean(grad_w):.4f}, std: {np.std(grad_w):.4f}")
        print(f"    grad_b: {grad_b:.4f}")
        
        # Check if gradient is consistently pushing weights down
        if np.mean(grad_w) > 0:
            print(f"  ⚠️  Gradient is POSITIVE → w will DECREASE")
        
        # Update
        agg.w -= lr * grad_w
        agg.b -= lr * grad_b
    
    print("\n" + "="*70)
    print("FINAL STATE")
    print("="*70)
    print(f"Weights: {agg.w}")
    print(f"Weight sum: {np.sum(agg.w):.3f}")
    print(f"Bias: {agg.b:.3f}")
    
    if np.std(agg.w) < 0.01:
        print("\n❌ WEIGHTS COLLAPSED!")
    else:
        print("\n✅ Weights remain healthy")


def analyze_imbalance_effect():
    """Compare behavior with different positive rates."""
    print("\n\n" + "="*70)
    print("IMBALANCE EFFECT ANALYSIS")
    print("="*70)
    
    np.random.seed(42)
    m, n = 5, 100
    
    for pos_rate in [0.01, 0.05, 0.10, 0.30, 0.50]:
        print(f"\n{'='*70}")
        print(f"Positive rate: {pos_rate:.0%}")
        print(f"{'='*70}")
        
        y_true = (np.random.rand(n) < pos_rate).astype(float)
        
        # Good predictions
        R_hat = np.zeros((m, n))
        for u in range(m):
            for i in range(n):
                if y_true[i] == 1:
                    R_hat[u, i] = np.random.uniform(0.7, 0.9)
                else:
                    R_hat[u, i] = np.random.uniform(0.1, 0.3)
        
        # Train for 20 iterations
        agg = WeightedAggregator(m, init_uniform=True)
        lr = 0.1
        
        initial_w_sum = np.sum(agg.w)
        
        for _ in range(20):
            y_pred = agg.predict(R_hat)
            residual = y_pred - y_true
            grad_w = (R_hat @ residual) / len(residual)
            grad_b = np.mean(residual)
            agg.w -= lr * grad_w
            agg.b -= lr * grad_b
        
        final_w_sum = np.sum(agg.w)
        w_change = final_w_sum - initial_w_sum
        
        print(f"  Initial weight sum: {initial_w_sum:.3f}")
        print(f"  Final weight sum: {final_w_sum:.3f}")
        print(f"  Change: {w_change:+.3f}")
        print(f"  Final weights: {agg.w.round(3)}")
        print(f"  Bias: {agg.b:.3f}")
        
        if abs(final_w_sum) < 0.1:
            print("  ❌ COLLAPSED!")
        elif w_change < -0.5:
            print("  ⚠️  Shrinking significantly")
        else:
            print("  ✅ Stable")


def test_gradient_formula():
    """Verify gradient formula is correct."""
    print("\n\n" + "="*70)
    print("GRADIENT FORMULA VERIFICATION")
    print("="*70)
    
    # Create simple case where we know the answer
    m, n = 3, 10
    R_hat = np.array([
        [0.8, 0.7, 0.2, 0.1, 0.3, 0.9, 0.8, 0.2, 0.1, 0.3],
        [0.9, 0.8, 0.1, 0.2, 0.2, 0.8, 0.9, 0.3, 0.1, 0.2],
        [0.7, 0.6, 0.3, 0.2, 0.4, 0.7, 0.7, 0.1, 0.2, 0.4],
    ])
    y_true = np.array([1., 1., 0., 0., 0., 1., 1., 0., 0., 0.])
    
    agg = WeightedAggregator(m, init_uniform=True)
    
    # Manual gradient computation
    y_pred = agg.predict(R_hat)
    residual = y_pred - y_true
    
    print("\nManual computation:")
    print(f"  y_pred: {y_pred.round(3)}")
    print(f"  y_true: {y_true}")
    print(f"  residual: {residual.round(3)}")
    
    # Gradient w.r.t. w[0] (first classifier)
    # ∂L/∂w[0] = (1/n) Σ (y_pred - y_true) * R_hat[0, i]
    grad_w0_manual = np.mean(residual * R_hat[0, :])
    
    # Gradient via matrix multiplication
    grad_w = (R_hat @ residual) / len(residual)
    
    print(f"\nGradient for w[0]:")
    print(f"  Manual: {grad_w0_manual:.4f}")
    print(f"  Matrix: {grad_w[0]:.4f}")
    print(f"  Match: {np.isclose(grad_w0_manual, grad_w[0])}")
    
    # Check all
    for i in range(m):
        manual = np.mean(residual * R_hat[i, :])
        assert np.isclose(manual, grad_w[i]), f"Mismatch for w[{i}]!"
    
    print("\n✅ Gradient formula is correct!")


if __name__ == '__main__':
    analyze_gradient_behavior()
    analyze_imbalance_effect()
    test_gradient_formula()
