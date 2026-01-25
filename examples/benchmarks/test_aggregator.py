"""
Minimal test to check if aggregator learns properly.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.ensemble.aggregators import WeightedAggregator

def test_aggregator_learning():
    """Test if weighted aggregator can learn from simple data."""
    print("="*60)
    print("TEST: Can WeightedAggregator learn?")
    print("="*60)
    
    # Create simple scenario
    np.random.seed(42)
    m, d, n = 5, 3, 100  # 5 classifiers, 3 latent dims, 100 instances
    
    # Create factors
    X = np.random.randn(d, m) * 0.5  # Classifier factors
    Y = np.random.randn(d, n) * 0.5  # Instance factors
    
    # Reconstruct probabilities
    R_hat = X.T @ Y  # (m × n)
    R_hat = 1 / (1 + np.exp(-R_hat))  # Sigmoid to [0,1]
    
    # Create labels: Make it so first latent dimension dominates
    # True label = whether first component of Y is positive
    y_true = (Y[0, :] > 0).astype(float)
    
    print(f"\nData:")
    print(f"  Classifiers: {m}")
    print(f"  Instances: {n}")
    print(f"  Positive rate: {y_true.mean():.2f}")
    
    # Test simple average baseline
    y_pred_mean = np.mean(R_hat, axis=0)
    from sklearn.metrics import roc_auc_score
    auc_mean = roc_auc_score(y_true, y_pred_mean)
    print(f"\nSimple Average AUC: {auc_mean:.3f}")
    
    # Initialize aggregator
    agg = WeightedAggregator(n_classifiers=m, init_uniform=True)
    print(f"\nInitial weights: {agg.w}")
    print(f"Initial bias: {agg.b:.4f}")
    
    # Test initial predictions
    y_pred_init = agg.predict(R_hat)
    auc_init = roc_auc_score(y_true, y_pred_init)
    print(f"Initial AUC: {auc_init:.3f}")
    
    # Compute initial loss
    from sklearn.metrics import log_loss
    loss_init = log_loss(y_true, np.clip(y_pred_init, 1e-7, 1-1e-7))
    print(f"Initial Loss: {loss_init:.4f}")
    
    # Train aggregator for several iterations
    print("\nTraining aggregator...")
    labeled_idx = np.ones(n, dtype=bool)  # All labeled
    
    for i in range(20):
        agg.update(X, Y, labeled_idx, y_true, lr=0.1)
        
        if i % 5 == 0:
            y_pred = agg.predict(R_hat)
            auc = roc_auc_score(y_true, y_pred)
            loss = log_loss(y_true, np.clip(y_pred, 1e-7, 1-1e-7))
            print(f"  Iter {i:2d}: Loss={loss:.4f}, AUC={auc:.3f}, w={agg.w.round(3)}, b={agg.b:.3f}")
    
    # Final evaluation
    y_pred_final = agg.predict(R_hat)
    auc_final = roc_auc_score(y_true, y_pred_final)
    loss_final = log_loss(y_true, np.clip(y_pred_final, 1e-7, 1-1e-7))
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Initial Loss: {loss_init:.4f} → Final Loss: {loss_final:.4f}")
    print(f"  Initial AUC: {auc_init:.3f} → Final AUC: {auc_final:.3f}")
    print(f"  Simple Average AUC: {auc_mean:.3f}")
    print(f"\nFinal weights: {agg.w}")
    print(f"Final bias: {agg.b:.4f}")
    
    if loss_final < loss_init - 0.01:
        print("\n✓ Aggregator IS learning (loss decreased)")
    else:
        print("\n❌ Aggregator NOT learning (loss didn't decrease)")
    
    if auc_final > auc_mean + 0.01:
        print("✓ Aggregator better than simple average")
    else:
        print("⚠️  Aggregator not better than simple average")


if __name__ == '__main__':
    test_aggregator_learning()
