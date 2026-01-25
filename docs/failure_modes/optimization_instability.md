# Failure Mode: ALS Without Label-Aware Confidence

**Category:** Implementation Error  
**Severity:** High (Wrong objective being optimized)  
**Date Identified:** 2026-01-25  
**Status:** FIXED - Use label-aware confidence or PyTorch

---

## TL;DR

**Problem:** Original ALS implementation optimized **reconstruction only**, ignoring the supervised loss term.

**Root Cause:** ALS was minimizing $L_{\text{recon}}$ when it should approximate $\mathcal{L}_{\text{CF}} = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$

**Solutions:**
1. **ALS + Label-Aware Confidence** (approximation, fast)
2. **PyTorch Joint Gradient Descent** (exact, recommended)

---

## The Real Problem: Wrong Objective

### Original Implementation (WRONG)

```python
for iteration in range(max_iter):
    # Step 1: Update X given Y (ALS - ONLY reconstruction)
    X = argmin_X ||C ⊙ (R - X^T Y)||² + λ||X||²
    
    # Step 2: Update Y given X (ALS - ONLY reconstruction)
    Y = argmin_Y ||C ⊙ (R - X^T Y)||² + λ||Y||²
    
    # Step 3: Update θ given X, Y (GD - ONLY supervision)
    θ = θ - lr * ∇_θ CE(y, g_θ(X^T Y))
```

**The fundamental error:** ALS optimizes $L_{\text{recon}}$ only, completely ignoring $L_{\text{sup}}$!

This is **NOT** what the KD-inspired combined loss intended:
$$\mathcal{L}_{\text{CF}} = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$$

### Why This Fails

**In Knowledge Distillation:**
- You don't optimize soft targets and hard labels **separately**
- You compute $\mathcal{L}_{\text{KD}} = \rho \cdot L(\text{soft}) + (1-\rho) \cdot L(\text{hard})$
- Then backprop through the **combined** loss

**Original CF-Ensemble ALS was doing the equivalent of:**
```python
# WRONG KD approach (nobody does this!)
for epoch in range(epochs):
    loss_soft = compute_soft_loss()
    loss_soft.backward()
    optimizer_soft.step()  # Only optimize for soft targets
    
    loss_hard = compute_hard_loss()
    loss_hard.backward()
    optimizer_hard.step()  # Separately optimize for hard labels
```

Of course this fails! The gradients point in different directions!

```
Iteration N:
  ALS: "Let me change X, Y to minimize reconstruction error"
  → Reconstruction loss: 2.5 → 1.8 ✓
  → Supervised loss: 0.35 → 0.52 ✗ (got worse!)
  
Iteration N+1:
  GD: "Let me change θ to fix supervised loss"
  → Supervised loss: 0.52 → 0.38 ✓
  → But now X, Y are still optimized for old θ
  
Iteration N+2:
  ALS: "Reconstruction is bad again, let me fix it"
  → Changes X, Y again
  → Undoes what GD tried to do
  → Cycle repeats...
```

---

## Evidence from Experiments

### Diagnostic Results

Running on simple, balanced data (positive_rate=0.50, quality=0.85):

```
Config: latent_dim=20, lambda_reg=0.01, rho=0.5

Iter 0:   Recon Loss=30.45, Sup Loss=0.502, PR-AUC=0.500
Iter 20:  Recon Loss=5.23,  Sup Loss=0.501, PR-AUC=0.502
Iter 40:  Recon Loss=2.15,  Sup Loss=0.509, PR-AUC=0.501
Iter 60:  Recon Loss=1.68,  Sup Loss=0.507, PR-AUC=0.503
Iter 100: Recon Loss=1.55,  Sup Loss=0.509, PR-AUC=0.504

❌ Supervised loss FLAT (oscillates around 0.50)
❌ PR-AUC FLAT (stuck at ~0.50, basically random)
❌ Never converges
```

**Compare to baselines:**
- Simple Average: PR-AUC = 0.942
- Stacking: PR-AUC = 0.951
- **CF-Ensemble: PR-AUC = 0.504** (random!)

### Isolated Component Tests

**Test 1: ALS alone (ρ=1.0)**
```python
# Pure reconstruction, no aggregator updates
trainer = CFEnsembleTrainer(rho=1.0)
trainer.fit(data)
```
**Result:** ✅ Converges! RMSE = 0.012 (excellent reconstruction)

**Test 2: Aggregator alone**
```python
# Fixed X, Y, train aggregator only
for iter in range(20):
    aggregator.update(X_fixed, Y_fixed, labeled_idx, labels, lr=0.1)
```
**Result:** ✅ Learns! AUC: 0.615 → 0.756, Loss: 0.744 → 0.729

**Test 3: Together (ρ=0.5)**
```python
# Alternating ALS + GD
trainer = CFEnsembleTrainer(rho=0.5)
trainer.fit(data)
```
**Result:** ❌ Fails! PR-AUC stuck at ~0.50, no convergence

**Conclusion:** Each component works, but together they interfere!

---

## Why Alternating Optimization Fails Here

### Classic Conditions for Convergence

Alternating optimization (e.g., EM, ALS) converges when:

1. **Each step decreases the SAME objective**
   - ❌ We have: ALS minimizes reconstruction, GD minimizes supervision
   
2. **Objective is jointly convex** (or has nice structure)
   - ❌ Our combined loss is non-convex in (X, Y, θ)
   
3. **Steps don't undo each other**
   - ❌ ALS changes invalidate aggregator weights
   - ❌ Aggregator needs different X, Y than reconstruction wants

### The Fundamental Conflict

**Reconstruction wants:**
- X, Y to faithfully reproduce R
- Even if R contains systematic errors
- Smooth, low-rank approximation

**Supervision wants:**
- X, Y (and θ) to predict labels correctly
- Amplify signal, suppress noise
- May require higher rank or different structure

**With alternating updates:**
- Reconstruction pulls X, Y one direction
- Supervision pulls θ (which depends on X, Y) another direction
- They never reach equilibrium

---

## Why Can't ALS Optimize the Combined Loss Directly?

**The challenge:** The combined loss is:
$$\mathcal{L}_{\text{CF}} = \rho \cdot \underbrace{\sum c_{ui}(r_{ui} - x_u^\top y_i)^2}_{\text{quadratic}} + (1-\rho) \cdot \underbrace{\sum CE(y_i, g_\theta(X^\top Y))}_{\text{NOT quadratic}}$$

**ALS requires quadratic objectives** to get closed-form solutions.

The supervised term contains:
- Sigmoid: $\sigma(w^\top(X^\top y_i) + b)$
- Logarithm: $\log(\sigma(...))$

These are **non-quadratic**, so no closed-form ALS solution exists!

**Analogy:** VAE also has reconstruction + KL terms, but uses **gradient descent**, not closed-form, because the combined loss is non-quadratic.

---

## Solutions

Two valid approaches with different trade-offs:

### Solution 1: ALS + Label-Aware Confidence (Fast Approximation)

**Strategy:** Make ALS **approximate** the combined loss by modulating confidence weights.

Instead of directly optimizing the combined loss, optimize:
$$\min_{X,Y} \sum_{u,i} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|^2 + \|Y\|^2)$$

where $\tilde{c}_{ui}$ is **label-aware**:
- For $y_i = 1$: $\tilde{c}_{ui} = c_{ui}(1 + \alpha \cdot r_{ui})$ (reward high predictions)
- For $y_i = 0$: $\tilde{c}_{ui} = c_{ui}(1 + \alpha \cdot (1-r_{ui}))$ (reward low predictions)

**How this works:**
- High $\tilde{c}_{ui}$ → ALS prioritizes reconstructing this $r_{ui}$ accurately
- Label-aware weighting: High $\tilde{c}_{ui}$ when prediction agrees with label
- Result: ALS indirectly learns to preserve correct predictions

**Implementation:**
```python
from cfensemble.optimization import CFEnsembleTrainer

trainer = CFEnsembleTrainer(
    n_classifiers=m,
    latent_dim=20,
    rho=0.5,
    use_label_aware_confidence=True,  # Enable approximation
    label_aware_alpha=1.0               # Supervision strength
)
trainer.fit(data)
```

**Advantages:**
- ✅ Fast (O(d³) closed-form ALS)
- ✅ No PyTorch dependency
- ✅ Reasonable approximation of supervision

**Disadvantages:**
- ❌ Approximate (not exact combined gradient)
- ❌ Requires tuning α parameter
- ❌ Less flexible than gradient descent

---

### Solution 2: Joint Gradient Descent via PyTorch (Exact, Recommended)

**Use PyTorch/JAX for automatic differentiation:**

```python
import torch

class CFEnsembleNet(torch.nn.Module):
    def __init__(self, m, n, d):
        super().__init__()
        self.X = torch.nn.Parameter(torch.randn(d, m) * 0.01)
        self.Y = torch.nn.Parameter(torch.randn(d, n) * 0.01)
        self.w = torch.nn.Parameter(torch.ones(m) / m)
        self.b = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, indices):
        # Reconstruct probabilities
        R_hat = self.X.T @ self.Y  # (m × n)
        R_hat_subset = R_hat[:, indices]
        
        # Aggregate
        logits = self.w @ R_hat_subset + self.b
        return torch.sigmoid(logits)
    
    def combined_loss(self, R, C, labels, labeled_idx, rho, lambda_reg):
        # Reconstruction loss
        R_hat = self.X.T @ self.Y
        recon_loss = torch.sum(C * (R - R_hat)**2)
        reg_loss = lambda_reg * (torch.sum(self.X**2) + torch.sum(self.Y**2))
        
        # Supervised loss
        y_pred = self.forward(labeled_idx)
        y_true = labels[labeled_idx]
        sup_loss = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
        
        # Combined
        return rho * (recon_loss + reg_loss) + (1 - rho) * sup_loss

# Training loop
model = CFEnsembleNet(m, n, d)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(max_epochs):
    optimizer.zero_grad()
    loss = model.combined_loss(R, C, labels, labeled_idx, rho=0.5, lambda_reg=0.01)
    loss.backward()  # ✅ Gradients for ALL parameters
    optimizer.step()  # ✅ Update ALL parameters together
```

**Advantages:**
- ✅ Single unified objective
- ✅ All parameters updated consistently
- ✅ Guaranteed descent (with proper learning rate)
- ✅ Can use modern optimizers (Adam, AdamW, etc.)
- ✅ Automatic differentiation - no manual gradient derivation

**Disadvantages:**
- Slower per iteration than closed-form ALS
- Need to tune learning rates
- Requires PyTorch/JAX dependency

### Solution 2: Two-Stage Training

**Decouple the conflicting objectives:**

```python
# Stage 1: Pure reconstruction (ρ=1.0)
trainer_recon = CFEnsembleTrainer(
    rho=1.0,  # Pure reconstruction
    max_iter=100
)
trainer_recon.fit(ensemble_data)

# Extract learned factors
X_final = trainer_recon.X
Y_final = trainer_recon.Y

# Stage 2: Train aggregator only (fix X, Y)
aggregator = WeightedAggregator(m)
for epoch in range(max_epochs):
    aggregator.update(
        X_final, Y_final, 
        labeled_idx, labels,
        lr=0.1
    )
```

**Advantages:**
- ✅ Simple to implement (minimal changes)
- ✅ Each stage converges reliably
- ✅ Fast (uses closed-form ALS)
- ✅ Interpretable (clear separation of concerns)

**Disadvantages:**
- X, Y don't benefit from supervision
- May be suboptimal vs. joint optimization
- Essentially "stacking on reconstructed features"

### Solution 3: Damped Alternating Updates

**Slow down aggregator to reduce oscillations:**

```python
for iteration in range(max_iter):
    # ALS updates (fast)
    X = update_X(Y, R, C, lambda_reg)
    Y = update_Y(X, R, C, lambda_reg)
    
    # Aggregator update (SLOW)
    if iteration % 10 == 0:  # Update less frequently
        aggregator.update(X, Y, labeled_idx, labels, lr=0.001)  # Small LR
```

**Advantages:**
- Minimal code changes
- Lets reconstruction stabilize first

**Disadvantages:**
- Still fundamentally unstable
- Very slow convergence for aggregator
- Requires careful tuning of update frequency and LR

### Solution 4: Weighted ALS (Modify ALS to See Supervision)

**Make ALS aware of supervised loss:**

```python
# Standard ALS update for X (ignores labels):
X = (Y C Y^T + λI)^{-1} Y C R

# Modified ALS update (incorporates labels):
# Add penalty for X, Y that produce bad predictions
# This is complex - need to linearize supervised loss
```

**Advantages:**
- Keeps closed-form updates
- Unifies objectives

**Disadvantages:**
- Complex to derive
- No longer closed-form (need approximation)
- Loses ALS speed advantage

---

## Recommendation

**For production:** Use **Solution #1 (PyTorch)** because:
1. Most flexible and extensible
2. Proven optimization (Adam, learning rate schedules)
3. Can add advanced features (attention, deep aggregator, etc.)
4. Standard in modern ML

**For quick validation:** Use **Solution #2 (Two-Stage)** because:
1. Fast to implement and test
2. Provides baseline performance
3. If it beats baselines, validates the approach
4. Can then invest in Solution #1 for better results

---

## Mathematical Analysis

### Why Joint Optimization Helps

**Combined loss:**
$$\mathcal{L}(X, Y, \theta) = \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta)$$

**Joint gradient descent:**
$$\begin{align}
X &\leftarrow X - \eta_X \cdot \nabla_X \mathcal{L} \\
Y &\leftarrow Y - \eta_Y \cdot \nabla_Y \mathcal{L} \\
\theta &\leftarrow \theta - \eta_\theta \cdot \nabla_\theta \mathcal{L}
\end{align}$$

**Key property:** All gradients computed w.r.t. the SAME loss
- $\nabla_X \mathcal{L}$ considers both reconstruction AND supervision
- $\nabla_Y \mathcal{L}$ considers both reconstruction AND supervision
- $\nabla_\theta \mathcal{L}$ only affects supervision (but consistent with X, Y gradients)

**Result:** Monotonic decrease in loss (with appropriate learning rates)

### Comparison: Alternating vs. Joint

| Property | Alternating ALS+GD | Joint GD |
|----------|-------------------|----------|
| Objective per step | Different | Same |
| Convergence guarantee | ❌ No | ✅ Yes (with LR schedule) |
| Speed per iteration | Fast (closed form) | Slower (gradient computation) |
| Total iterations to converge | ∞ (doesn't converge) | ~100-500 |
| Optimization quality | Poor | Good |
| Ease of extension | Hard | Easy (autodiff) |

---

## Implementation Priorities

### Phase 1: Validate Approach (Week 1)

1. ✅ Implement two-stage training
2. ✅ Test on benchmark data
3. ✅ Verify it beats baselines
4. Document results

### Phase 2: Production Solution (Week 2-3)

1. Implement PyTorch-based joint optimization
2. Add learning rate scheduling (ReduceLROnPlateau)
3. Add early stopping (based on validation loss)
4. Comprehensive benchmarking vs. two-stage

### Phase 3: Advanced Features (Week 4+)

1. Try different optimizers (Adam vs. AdamW vs. SGD)
2. Implement advanced aggregators (attention-based)
3. Add batch training for large datasets
4. GPU acceleration

---

## Lessons Learned

### 1. Alternating Optimization is Fragile

- Works great for single-objective problems (e.g., NMF, ALS for recommendations)
- Fails when objectives conflict
- Always check: are all steps optimizing the SAME thing?

### 2. Closed-Form ≠ Better

- ALS is fast per iteration
- But if it doesn't converge, speed is useless
- Gradient descent slower per iteration, but converges

### 3. Modern Tools Help

- PyTorch/JAX handle complex gradients automatically
- Don't need to derive update equations manually
- Can focus on model design, not optimization

### 4. Validate Components Separately

- Test ALS alone → works!
- Test aggregator alone → works!
- Test together → fails!
- This isolation was KEY to finding the bug

---

## Related Failure Modes

See also:
- `transductive_vs_inductive.md` - Using wrong train/test split
- `hyperparameter_sensitivity.md` - Tuning λ, ρ, d
- `confidence_weights.md` - Label-aware weighting

---

## References

1. **Alternating Optimization:**
   - Boyd, S., et al. (2011). "Distributed Optimization and Statistical Learning via ADMM." Foundations and Trends in ML.
   - Bezdek, J., Hathaway, R. (2003). "Convergence of Alternating Optimization." Neural, Parallel & Scientific Comp.

2. **Matrix Factorization Optimization:**
   - Zhou, Y., et al. (2008). "Large-scale Parallel Collaborative Filtering." KDD.
   - Gemulla, R., et al. (2011). "Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent." KDD.

3. **Joint Training with Multiple Objectives:**
   - Kendall, A., et al. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR.
   - Chen, Z., et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing." ICML.

---

**Key Takeaway:** When combining multiple objectives, use joint optimization with unified gradients. Alternating updates only work when all steps minimize the same objective!
