"""
Diagnose PyTorch Gradients

Check if PyTorch has the same class imbalance problem in supervised loss gradients.
"""

import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cfensemble.data import generate_imbalanced_ensemble_data, EnsembleData
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer, CFEnsembleNet


def diagnose_pytorch_gradients():
    """Trace PyTorch gradients during training."""
    
    print("="*80)
    print("PYTORCH GRADIENT DIAGNOSIS")
    print("="*80)
    
    # Generate data
    R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
        n_instances=500,
        n_classifiers=10,
        n_labeled=250,
        positive_rate=0.10,
        target_quality=0.70,
        random_state=42
    )
    
    n = R.shape[1]
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    print(f"\nData: {R.shape}")
    print(f"Labeled: {np.sum(labeled_mask)}")
    print(f"Positive rate: {np.mean(y_true[labeled_mask]):.1%}")
    
    # Convert to PyTorch
    device = torch.device('cpu')
    R_torch = torch.from_numpy(R).float().to(device)
    C_torch = torch.ones_like(R_torch)  # Uniform confidence
    labels_torch = torch.from_numpy(labels).float().to(device)
    labeled_mask_torch = torch.from_numpy(labeled_mask).bool().to(device)
    
    # Initialize model
    m, n = R.shape
    model = CFEnsembleNet(m=m, n=n, d=15, aggregator_type='weighted').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("\n" + "="*80)
    print("TRAINING TRACE (First 10 epochs)")
    print("="*80)
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Compute loss
        loss, loss_dict = model.compute_loss(
            R_torch, C_torch, labels_torch, labeled_mask_torch,
            rho=0.5, lambda_reg=0.01
        )
        
        # Backward
        loss.backward()
        
        # Check gradients
        w_grad = model.w.grad.detach().cpu().numpy()
        b_grad = model.b.grad.detach().cpu().numpy()
        
        w = model.w.detach().cpu().numpy()
        b = model.b.item()
        
        print(f"\nEpoch {epoch}:")
        print(f"  Weights: [{w[0]:.4f}, {w[1]:.4f}, ..., {w[-1]:.4f}]")
        print(f"  Weight sum: {np.sum(w):.4f}, std: {np.std(w):.4f}")
        print(f"  Bias: {b:.4f}")
        
        print(f"  Gradients (w): [{w_grad[0]:.4f}, {w_grad[1]:.4f}, ..., {w_grad[-1]:.4f}]")
        print(f"  Gradient mean: {np.mean(w_grad):.4f}")
        print(f"  Gradient (b): {b_grad[0]:.4f}")
        
        if np.mean(w_grad) < 0:
            print(f"  ⚠️  Gradient is NEGATIVE → w will INCREASE")
        else:
            print(f"  ⚠️  Gradient is POSITIVE → w will DECREASE")
        
        print(f"  Loss: total={loss_dict['total']:.4f}, "
              f"sup={loss_dict['supervised']:.4f}, "
              f"recon={loss_dict['reconstruction']:.4f}")
        
        # Update
        optimizer.step()
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Get final predictions
    model.eval()
    with torch.no_grad():
        R_hat = model.reconstruct()
        labeled_idx_torch = torch.where(labeled_mask_torch)[0]
        y_pred = model.forward(labeled_idx_torch)
        y_true_labeled = labels_torch[labeled_mask_torch]
        
        # Separate by class
        pos_mask = (y_true_labeled == 1)
        neg_mask = (y_true_labeled == 0)
        
        residual = y_pred - y_true_labeled
        
        print(f"\nPredictions on labeled data:")
        print(f"  Mean: {y_pred.mean().item():.4f}")
        print(f"  Std: {y_pred.std().item():.4f}")
        print(f"  On positives ({pos_mask.sum()} instances): {y_pred[pos_mask].mean().item():.4f}")
        print(f"  On negatives ({neg_mask.sum()} instances): {y_pred[neg_mask].mean().item():.4f}")
        
        print(f"\nResiduals:")
        print(f"  Mean: {residual.mean().item():.4f}")
        print(f"  On positives: {residual[pos_mask].mean().item():.4f}")
        print(f"  On negatives: {residual[neg_mask].mean().item():.4f}")
        
        # This is the key: check if negatives dominate
        print(f"\n🔍 KEY INSIGHT:")
        print(f"   Positive instances: {pos_mask.sum().item()} ({pos_mask.sum().item()/len(pos_mask)*100:.1f}%)")
        print(f"   Negative instances: {neg_mask.sum().item()} ({neg_mask.sum().item()/len(neg_mask)*100:.1f}%)")
        print(f"   Residual contribution from positives: {residual[pos_mask].sum().item():.4f}")
        print(f"   Residual contribution from negatives: {residual[neg_mask].sum().item():.4f}")
        
        total_residual = residual.sum().item()
        print(f"   Total residual: {total_residual:.4f}")
        
        if abs(residual[neg_mask].sum().item()) > abs(residual[pos_mask].sum().item()):
            print(f"\n   ❌ Negatives DOMINATE gradient (class imbalance problem!)")
        else:
            print(f"\n   ✅ Balanced gradient")


if __name__ == '__main__':
    diagnose_pytorch_gradients()
