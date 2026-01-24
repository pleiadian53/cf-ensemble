# ALS vs PyTorch Gradient Descent for CF-Ensemble

**Comparing two optimization approaches: Closed-form ALS vs Gradient-based PyTorch**

---

## Quick Answer

**Is ALS state-of-the-art?**  
✅ **Yes, for basic matrix factorization.** ALS is the gold standard for collaborative filtering because:
- Closed-form updates (no learning rate tuning)
- Guaranteed to decrease loss each iteration
- Well-studied, numerically stable
- Used in production systems (Spotify, Netflix, etc.)

**Should we also implement PyTorch?**  
✅ **Yes, for flexibility and future extensions.** PyTorch offers:
- GPU acceleration
- Modern optimizers (Adam, AdamW, etc.)
- Easy to add non-linearities, neural components
- Richer ecosystem for experimentation

**Bottom line**: They should give **equivalent results** for the basic objective, but each has unique advantages.

---

## Detailed Comparison

### Current Implementation: Alternating Least Squares (ALS)

**Algorithm**: For each iteration:
1. Fix Y, solve for X: $x_u = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u$
2. Fix X, solve for Y: $y_i = (X C_i X^T + \lambda I)^{-1} X C_i r_i$
3. Update aggregator via gradient descent

**Advantages**:
- ✅ **Closed-form solution**: No learning rate, no convergence issues
- ✅ **Guaranteed decrease**: Each ALS step reduces reconstruction loss
- ✅ **Numerically stable**: Uses `np.linalg.solve` (stable for ill-conditioned systems)
- ✅ **Fast for small-medium problems**: O(md³) for X, O(nd³) for Y
- ✅ **Well-understood**: Decades of research, known convergence properties
- ✅ **No hyperparameter tuning**: Just λ (regularization)

**Disadvantages**:
- ❌ **Not vectorized**: Loops over m classifiers and n instances
- ❌ **No GPU acceleration**: NumPy is CPU-only
- ❌ **Hard to extend**: Closed-form limits to linear factorization
- ❌ **Slow for huge datasets**: O(nd³) becomes expensive when n > 100,000

**When to use**:
- Small to medium datasets (m, n < 10,000)
- CPU-only environment
- Want stable, "set and forget" training
- Research/prototyping phase

---

### Alternative: PyTorch Gradient Descent

**Algorithm**: For each iteration:
1. Compute loss: $L = \rho \sum c_{ui}(r_{ui} - x_u^T y_i)^2 + (1-\rho) \sum CE(y_i, g(r̂_i)) + \lambda(||X||^2 + ||Y||^2)$
2. Backpropagate: $\frac{\partial L}{\partial X}, \frac{\partial L}{\partial Y}, \frac{\partial L}{\partial \theta}$
3. Update all parameters: $X \leftarrow X - \eta \nabla_X L$ (and Y, θ)

**Advantages**:
- ✅ **GPU acceleration**: 10-100x speedup for large problems
- ✅ **Vectorized**: Batched operations, no Python loops
- ✅ **Modern optimizers**: Adam, AdamW (adaptive learning rates)
- ✅ **Easy to extend**: Add neural networks, attention, non-linearities
- ✅ **Unified framework**: All parameters updated together
- ✅ **Scalable**: Works for n > 1 million

**Disadvantages**:
- ❌ **Learning rate tuning**: Need to find good η
- ❌ **Can diverge**: Gradient descent not guaranteed to converge
- ❌ **More hyperparameters**: η, optimizer choice, batch size, etc.
- ❌ **Overkill for simple problems**: More complex than needed

**When to use**:
- Large datasets (n > 10,000)
- GPU available
- Want to experiment with extensions (neural aggregators, etc.)
- Production deployment with high throughput needs

---

## Mathematical Equivalence

For the **basic matrix factorization objective**, ALS and PyTorch should converge to **similar solutions**:

$$L_{\text{recon}} = \sum_{u,i} c_{ui}(r_{ui} - x_u^T y_i)^2 + \lambda(||X||^2 + ||Y||^2)$$

**Why equivalent?**
- Both optimize the same objective
- Both are guaranteed to find a local minimum (under convexity assumptions)
- Regularization is identical

**Expected differences**:
- **Convergence path**: ALS alternates, PyTorch updates jointly → Different trajectories
- **Speed**: ALS may be faster for small problems, PyTorch faster for large
- **Final loss**: Should be within ~0.1% of each other
- **Predictions**: Should have correlation > 0.99

---

## Implementation: PyTorch Version

Here's a sketch of how to implement CF-Ensemble in PyTorch:

### Basic PyTorch Trainer

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchCFEnsemble(nn.Module):
    """PyTorch implementation of CF-Ensemble."""
    
    def __init__(self, n_classifiers, n_instances, latent_dim=20, rho=0.5):
        super().__init__()
        
        # Latent factors as learnable parameters
        self.X = nn.Parameter(torch.randn(latent_dim, n_classifiers) * 0.01)
        self.Y = nn.Parameter(torch.randn(latent_dim, n_instances) * 0.01)
        
        # Aggregator weights
        self.agg_weights = nn.Parameter(torch.ones(n_classifiers) / n_classifiers)
        self.agg_bias = nn.Parameter(torch.zeros(1))
        
        self.rho = rho
        self.latent_dim = latent_dim
    
    def forward(self, labeled_idx=None):
        """Reconstruct probability matrix."""
        R_hat = self.X.T @ self.Y  # (m × n)
        
        if labeled_idx is not None:
            # Aggregate for labeled instances
            R_hat_labeled = R_hat[:, labeled_idx]
            logits = self.agg_weights @ R_hat_labeled + self.agg_bias
            predictions = torch.sigmoid(logits)
            return R_hat, predictions
        
        return R_hat, None
    
    def compute_loss(self, R, C, labels, labeled_idx, lambda_reg):
        """Compute combined loss."""
        R_hat, predictions = self.forward(labeled_idx)
        
        # Reconstruction loss
        residuals = R - R_hat
        weighted_mse = torch.sum(C * residuals ** 2)
        reg_term = lambda_reg * (torch.sum(self.X ** 2) + torch.sum(self.Y ** 2))
        recon_loss = weighted_mse + reg_term
        
        # Supervised loss (if we have labels)
        if predictions is not None and len(labeled_idx) > 0:
            labels_tensor = labels[labeled_idx]
            bce = nn.BCELoss()
            sup_loss = bce(predictions, labels_tensor)
        else:
            sup_loss = torch.tensor(0.0)
        
        # Combined loss
        total_loss = self.rho * recon_loss + (1 - self.rho) * sup_loss
        
        return total_loss, recon_loss, sup_loss

def train_pytorch_cfensemble(R, labels, latent_dim=20, rho=0.5, 
                             lambda_reg=0.01, lr=0.01, max_iter=50):
    """Train CF-Ensemble using PyTorch."""
    m, n = R.shape
    labeled_idx = ~torch.isnan(labels)
    
    # Convert to tensors
    R_tensor = torch.FloatTensor(R)
    C_tensor = torch.FloatTensor(np.abs(R - 0.5))
    labels_tensor = torch.FloatTensor(labels)
    
    # Initialize model
    model = PyTorchCFEnsemble(m, n, latent_dim, rho)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {'loss': [], 'recon': [], 'sup': []}
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Compute loss
        total_loss, recon_loss, sup_loss = model.compute_loss(
            R_tensor, C_tensor, labels_tensor, labeled_idx, lambda_reg
        )
        
        # Backprop and update
        total_loss.backward()
        optimizer.step()
        
        # Log history
        history['loss'].append(total_loss.item())
        history['recon'].append(recon_loss.item())
        history['sup'].append(sup_loss.item())
        
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Loss={total_loss.item():.4f}")
    
    return model, history
```

### GPU-Accelerated Version

```python
def train_gpu(R, labels, device='cuda'):
    """Train on GPU."""
    R_gpu = torch.FloatTensor(R).to(device)
    labels_gpu = torch.FloatTensor(labels).to(device)
    
    model = PyTorchCFEnsemble(m, n, latent_dim, rho).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop (same as above, but on GPU)
    for iteration in range(max_iter):
        optimizer.zero_grad()
        loss, _, _ = model.compute_loss(R_gpu, C_gpu, labels_gpu, ...)
        loss.backward()
        optimizer.step()
    
    return model
```

**Expected speedup**: 10-50x for large datasets (n > 50,000)

---

## Comparison Experiment

### Recommended Test

```python
import numpy as np
from cfensemble.data import EnsembleData
from cfensemble.optimization import CFEnsembleTrainer

# Generate test data
np.random.seed(42)
m, n = 10, 1000
R = np.random.rand(m, n)
labels = np.random.randint(0, 2, n).astype(float)

# Train with ALS
print("Training with ALS...")
data = EnsembleData(R, labels)
als_trainer = CFEnsembleTrainer(
    n_classifiers=m,
    latent_dim=20,
    rho=0.5,
    lambda_reg=0.01,
    max_iter=50,
    random_seed=42
)
als_trainer.fit(data)
als_pred = als_trainer.predict(data)
als_loss = als_trainer.history['loss'][-1]

# Train with PyTorch
print("\nTraining with PyTorch...")
pt_model, pt_history = train_pytorch_cfensemble(
    R, labels,
    latent_dim=20,
    rho=0.5,
    lambda_reg=0.01,
    lr=0.01,  # May need tuning!
    max_iter=50
)
pt_pred = pt_model(torch.FloatTensor(R))[0].detach().numpy()
pt_pred_agg = pt_pred.mean(axis=0)  # Simple aggregation
pt_loss = pt_history['loss'][-1]

# Compare results
print("\n=== Comparison ===")
print(f"Final loss - ALS: {als_loss:.4f}, PyTorch: {pt_loss:.4f}")
print(f"Loss difference: {abs(als_loss - pt_loss):.4f}")

correlation = np.corrcoef(als_pred, pt_pred_agg)[0, 1]
print(f"Prediction correlation: {correlation:.4f}")

# Should be close!
assert correlation > 0.95, "Predictions should be highly correlated"
assert abs(als_loss - pt_loss) / als_loss < 0.1, "Losses should be within 10%"

print("\n✓ ALS and PyTorch give consistent results!")
```

**Expected output**:
```
Training with ALS...
Iter 0: Loss=0.4523, Recon=0.3012, Sup=0.6034
...
Iter 40: Loss=0.1234, Recon=0.0821, Sup=0.1647

Training with PyTorch...
Iter 0: Loss=0.4612
...
Iter 40: Loss=0.1289

=== Comparison ===
Final loss - ALS: 0.1234, PyTorch: 0.1289
Loss difference: 0.0055
Prediction correlation: 0.9823

✓ ALS and PyTorch give consistent results!
```

---

## When Each Approach is Better

### Use ALS when:
- ✅ Dataset size: m, n < 10,000
- ✅ Environment: CPU-only (laptops, small servers)
- ✅ Goal: Quick prototyping, research
- ✅ Priority: Stability over speed
- ✅ Expertise: Linear algebra >> Deep learning

### Use PyTorch when:
- ✅ Dataset size: n > 10,000 (especially n > 100,000)
- ✅ Environment: GPU available
- ✅ Goal: Production deployment, high throughput
- ✅ Priority: Scalability, extensibility
- ✅ Expertise: Deep learning, PyTorch ecosystem

### Use BOTH when:
- ✅ Research project: Compare convergence properties
- ✅ Validation: Ensure implementation correctness
- ✅ Transition: Start with ALS (fast iteration), deploy with PyTorch (scale)

---

## Extensions Only Possible with PyTorch

### 1. Neural Aggregators

```python
class NeuralAggregator(nn.Module):
    def __init__(self, n_classifiers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classifiers, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, r_hat):
        # r_hat: (m × n) → predictions: (n,)
        return self.net(r_hat.T).squeeze()
```

**Benefit**: Learn complex, non-linear aggregation patterns.

---

### 2. Attention-Based Weighting

```python
class AttentionAggregator(nn.Module):
    def __init__(self, n_classifiers):
        super().__init__()
        self.attention = nn.Linear(n_classifiers, n_classifiers)
    
    def forward(self, r_hat):
        # Compute instance-dependent weights
        attn_scores = torch.softmax(self.attention(r_hat.T), dim=1)
        weighted = (r_hat.T * attn_scores).sum(dim=1)
        return torch.sigmoid(weighted)
```

**Benefit**: Different instances use different classifier weights.

---

### 3. Deep Matrix Factorization

```python
class DeepCFEnsemble(nn.Module):
    def __init__(self, n_classifiers, n_instances, latent_dim=20):
        super().__init__()
        
        # Encoder for classifiers
        self.encoder_X = nn.Sequential(
            nn.Linear(n_classifiers, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Encoder for instances
        self.encoder_Y = nn.Sequential(
            nn.Linear(n_instances, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, R):
        X = self.encoder_X(R.T).T  # (latent_dim × n_classifiers)
        Y = self.encoder_Y(R)       # (latent_dim × n_instances)
        R_hat = X.T @ Y
        return R_hat
```

**Benefit**: Non-linear latent representations.

---

## Hybrid Approach (Best of Both Worlds)

### Strategy: ALS for Initialization, PyTorch for Fine-Tuning

```python
# Phase 1: Quick convergence with ALS
als_trainer = CFEnsembleTrainer(n_classifiers=m, max_iter=20)
als_trainer.fit(data)

# Phase 2: Fine-tune with PyTorch (starting from ALS solution)
pt_model = PyTorchCFEnsemble(m, n, latent_dim=20)
pt_model.X.data = torch.FloatTensor(als_trainer.X)
pt_model.Y.data = torch.FloatTensor(als_trainer.Y)

# Fine-tune for a few more iterations
optimizer = optim.Adam(pt_model.parameters(), lr=0.001)  # Small LR
for _ in range(10):
    # ... training loop ...
```

**Benefit**:
- ALS gives good initialization (fast, stable)
- PyTorch refines solution (flexible, GPU-accelerated)

---

## Performance Benchmarks (Expected)

### Small Dataset (m=10, n=1,000)
| Method | Time | Final Loss | GPU? |
|--------|------|------------|------|
| ALS (NumPy) | 0.5s | 0.1234 | No |
| PyTorch (CPU) | 1.2s | 0.1289 | No |
| PyTorch (GPU) | 0.8s | 0.1289 | Yes |

**Winner**: ALS (fastest, simplest)

---

### Medium Dataset (m=20, n=10,000)
| Method | Time | Final Loss | GPU? |
|--------|------|------------|------|
| ALS (NumPy) | 15s | 0.2456 | No |
| PyTorch (CPU) | 25s | 0.2501 | No |
| PyTorch (GPU) | 2s | 0.2501 | **Yes** |

**Winner**: PyTorch GPU (10x faster)

---

### Large Dataset (m=50, n=100,000)
| Method | Time | Final Loss | GPU? |
|--------|------|------------|------|
| ALS (NumPy) | 600s | 0.3123 | No |
| PyTorch (CPU) | 1200s | 0.3187 | No |
| PyTorch (GPU) | 20s | 0.3187 | **Yes** |

**Winner**: PyTorch GPU (30x faster!)

---

## Recommendation

### For this Project

**Phase 1-2 (Current)**: ✅ **Use ALS**
- Already implemented and tested
- Perfect for development and validation
- Fast enough for typical datasets

**Phase 4-5 (Future)**: Consider **adding PyTorch**
- Useful for large-scale experiments
- Enables neural extensions (Phase 6+)
- Good for production deployment

**Implementation Plan**:
1. ✅ Phase 1-2: ALS (done!)
2. Phase 3: Confidence strategies (stick with ALS)
3. Phase 4: First experiments (ALS sufficient)
4. **Phase 5**: Add PyTorch implementation as alternative
5. **Phase 6**: Neural extensions (requires PyTorch)

---

## Would You Like Me to Implement PyTorch Version?

I can create a PyTorch implementation that:
- ✅ Matches the ALS API (drop-in replacement)
- ✅ Includes comparison experiments
- ✅ Validates consistency with ALS
- ✅ Adds GPU support
- ✅ Prepares for neural extensions

**Estimated effort**: ~2-3 hours (1 module, tests, comparison script)

**Benefits**:
- Validation that our ALS is correct (should match PyTorch)
- Future-proofs the codebase
- Enables GPU acceleration
- Opens door to neural aggregators

**When to do it**:
- After Phase 3 (confidence strategies)
- Before large-scale experiments (Phase 5)
- Or now, if you're excited about it! 😄

---

## References

- **ALS for Matrix Factorization**: Hu et al., "Collaborative Filtering for Implicit Feedback Datasets" (2008)
- **Deep Matrix Factorization**: He et al., "Neural Collaborative Filtering" (2017)
- **PyTorch for RecSys**: "Deep Learning Recommendation Models" (Facebook, 2019)

---

**Summary**:
- **ALS**: State-of-the-art for basic matrix factorization, perfect for Phase 1-3
- **PyTorch**: More flexible, scalable, enables future extensions
- **Both are valid**: Use ALS now, add PyTorch later for scale/flexibility
- **They should agree**: Correlation > 0.95, loss within 10%

---

*Both approaches are "correct" - choose based on your constraints and goals!*
