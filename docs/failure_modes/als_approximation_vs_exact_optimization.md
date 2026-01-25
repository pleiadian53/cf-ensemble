# ALS Approximation vs. Exact PyTorch Optimization

**Category:** Implementation Choice  
**Status:** Both valid approaches with different trade-offs  
**Date:** 2026-01-25

---

## TL;DR

**Goal:** Optimize the combined KD-inspired loss:
$$\mathcal{L}_{\text{CF}} = \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta)$$

**Two approaches:**
1. **ALS with label-aware confidence** (APPROXIMATION) - Fast but approximate
2. **PyTorch joint gradient descent** (EXACT) - Slower but exact

Neither is "wrong" - they're different algorithmic choices with trade-offs.

---

## The Core Challenge

### Why Can't ALS Optimize the Full Loss?

**ALS works for quadratic objectives:**
$$\min_{X, Y} \sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|^2 + \|Y\|^2)$$

This has **closed-form solutions**:
- Fix Y, solve for X: $(YC_uY^\top + \lambda I)x_u = YC_u r_u$
- Fix X, solve for Y: $(XC_iX^\top + \lambda I)y_i = XC_i r_i$

**The supervised loss is NOT quadratic:**
$$L_{\text{sup}} = \sum_{i \in \mathcal{L}} \underbrace{-y_i \log \sigma(w^\top(X^\top y_i) + b)}_{\text{non-quadratic!}} - (1-y_i) \log(1-\sigma(...))$$

- Contains $\sigma(\cdot)$ (sigmoid)
- Contains $\log(\cdot)$  
- No closed-form solution for $\nabla_X, \nabla_Y$

**Conclusion:** Cannot derive closed-form ALS for the full combined loss.

---

## Approach 1: ALS with Label-Aware Confidence (Approximation)

### Strategy

**Key insight:** Modulate confidence weights to incorporate supervision indirectly.

Instead of directly optimizing:
$$\mathcal{L}_{\text{CF}} = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$$

Approximate by optimizing:
$$\mathcal{L}_{\text{approx}} = \sum_{u,i} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|^2 + \|Y\|^2)$$

where $\tilde{c}_{ui}$ is **label-aware**:

```python
# For labeled instances:
if y_i == 1:
    c_ui = base_confidence * (1 + α * r_ui)       # Reward high predictions
else:
    c_ui = base_confidence * (1 + α * (1 - r_ui))  # Reward low predictions

# For unlabeled instances:
c_ui = base_confidence  # Typically |r_ui - 0.5| (certainty)
```

### How This Approximates Supervision

**Intuition:**
- High $c_{ui}$ → ALS prioritizes matching $r_{ui}$ during reconstruction
- Label-aware weighting: High $c_{ui}$ when prediction agrees with label
- Result: ALS "wants" to preserve correct predictions, discard errors

**Mathematical connection:**
$$\min_X \sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 \approx \min_X \left[\text{recon} + \alpha \cdot \text{supervision signal}\right]$$

The label-aware weighting creates an **implicit supervision gradient**.

### Implementation

```python
from cfensemble.data import EnsembleData
from cfensemble.optimization import CFEnsembleTrainer

# Create data
data = EnsembleData(R, labels)

# Train with label-aware confidence
trainer = CFEnsembleTrainer(
    n_classifiers=m,
    latent_dim=20,
    rho=0.5,
    use_label_aware_confidence=True,   # Enable approximation
    label_aware_alpha=1.0               # Supervision strength
)
trainer.fit(data)
```

### Trade-offs

**Advantages:**
- ✅ **Fast**: O(d³) closed-form ALS updates
- ✅ **No autodiff**: Works without PyTorch/JAX
- ✅ **Interpretable**: Confidence weights show which predictions matter
- ✅ **Scalable**: Parallelizable across classifiers/instances

**Disadvantages:**
- ❌ **Approximate**: Not exact gradient of combined loss
- ❌ **Indirect supervision**: α parameter needs tuning
- ❌ **Potential instability**: Alternating updates can oscillate
- ❌ **Limited flexibility**: Hard to extend (e.g., attention aggregators)

---

## Approach 2: PyTorch Joint Gradient Descent (Exact)

### Strategy

Directly optimize the combined loss via backpropagation:

```python
# Forward pass
R_hat = X.T @ Y
y_pred = aggregator(R_hat)

# Combined loss (exact)
loss = rho * reconstruction_loss(R, R_hat, C) + (1 - rho) * supervised_loss(y, y_pred)

# Backward pass (unified gradients)
loss.backward()  # Computes ∇_X L_CF, ∇_Y L_CF, ∇_θ L_CF

# Update (all parameters together)
optimizer.step()  # X, Y, θ updated consistently
```

### Why This is Exact

**Key property:** All gradients computed w.r.t. **same unified loss**

$$\begin{align}
\nabla_X \mathcal{L}_{\text{CF}} &= \rho \cdot \nabla_X L_{\text{recon}} + (1-\rho) \cdot \nabla_X L_{\text{sup}} \\
\nabla_Y \mathcal{L}_{\text{CF}} &= \rho \cdot \nabla_Y L_{\text{recon}} + (1-\rho) \cdot \nabla_Y L_{\text{sup}} \\
\nabla_\theta \mathcal{L}_{\text{CF}} &= (1-\rho) \cdot \nabla_\theta L_{\text{sup}}
\end{align}$$

- Each gradient considers **both** reconstruction AND supervision
- Updates move in direction that decreases **total** loss
- Guaranteed descent (with appropriate learning rate)

### Implementation

```python
from cfensemble.optimization import CFEnsemblePyTorchTrainer

# Train with exact optimization
trainer = CFEnsemblePyTorchTrainer(
    n_classifiers=m,
    latent_dim=20,
    rho=0.5,
    max_epochs=200,
    optimizer='adam',
    lr=0.01
)
trainer.fit(data)
```

### Trade-offs

**Advantages:**
- ✅ **Exact**: True gradient of combined loss
- ✅ **Unified**: All parameters updated consistently
- ✅ **Flexible**: Easy to extend (deep aggregators, attention, etc.)
- ✅ **Modern**: Standard approach in deep learning (VAE, multi-task, etc.)
- ✅ **GPU acceleration**: Can scale to large problems

**Disadvantages:**
- ❌ **Slower per iteration**: Gradient computation vs. closed-form
- ❌ **Requires autodiff**: PyTorch/JAX dependency
- ❌ **More hyperparameters**: Learning rates, optimizers, schedules

---

## The Knowledge Distillation Analogy

### How is KD Actually Optimized?

**Loss:**
$$\mathcal{L}_{\text{KD}} = \rho \cdot \underbrace{KL(q_{\text{teacher}} \| q_{\text{student}})}_{\text{soft targets}} + (1-\rho) \cdot \underbrace{CE(y, q_{\text{student}})}_{\text{hard labels}}$$

**Optimization:**
```python
# Nobody uses closed-form for KD!
loss = rho * soft_loss + (1 - rho) * hard_loss
loss.backward()
optimizer.step()  # Standard gradient descent
```

**Why?** Because:
- KL divergence is not quadratic
- Cross-entropy is not quadratic
- No closed-form solution exists

**CF-Ensemble is the same:** Combined loss requires gradient descent, not closed-form.

---

## The VAE Analogy

**VAE loss:**
$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}[\log p(x|z)]}_{\text{reconstruction}} + \underbrace{KL(q(z|x) \| p(z))}_{\text{prior regularization}}$$

**Optimization:**
- Reconstruction: $\log p(x|z)$ (not quadratic due to likelihood)
- KL term: Closed-form for Gaussian, but combined loss still needs gradients
- **Solution:** Reparameterization trick + backprop (not closed-form)

**Key insight:** Even with some closed-form components, the **combined** objective usually requires gradient descent.

---

## When to Use Which Approach

### Use ALS (Approximation) When:

✅ **Speed is critical**
- Need fast iterations (O(d³) vs. O(d²n) gradients)
- Large-scale problems where closed-form helps

✅ **No autodiff framework**
- Production environment without PyTorch/JAX
- Minimal dependencies required

✅ **Interpretability matters**
- Want to analyze confidence weights
- Need to explain which predictions were prioritized

✅ **Good enough approximation**
- Label-aware confidence captures supervision signal adequately
- Results competitive with exact optimization

**Example use cases:**
- Real-time systems needing <100ms inference
- Embedded systems with limited compute
- Exploratory analysis where speed matters

### Use PyTorch (Exact) When:

✅ **Accuracy is critical**
- Need best possible performance
- Research/publication requiring exact optimization

✅ **Complex models**
- Advanced aggregators (attention, transformers)
- Deep architectures beyond simple weighted average

✅ **GPU available**
- Can leverage hardware acceleration
- Large batch training

✅ **Standard ML pipeline**
- Already using PyTorch for other models
- Want consistency with modern ML practices

**Example use cases:**
- Production ML systems with GPU infrastructure
- Research exploring new aggregation architectures
- Scenarios where PyTorch already in stack

---

## Empirical Comparison

### Expected Performance (Hypothesis)

Based on approximation theory:

| Metric | ALS + Label-Aware | PyTorch Joint |
|--------|------------------|---------------|
| PR-AUC | 0.30-0.40 | **0.35-0.45** |
| Convergence | 50-200 iter | **30-100 epochs** |
| Speed (CPU) | **1-2 sec** | 5-10 sec |
| Speed (GPU) | N/A | **1-2 sec** |
| Memory | **Low** | Medium |

**Prediction:** PyTorch should be 5-15% better PR-AUC, ALS should be 2-5x faster on CPU.

### Benchmark Plan

Run `pytorch_vs_als_benchmark.py` with both methods:
```bash
python examples/benchmarks/pytorch_vs_als_benchmark.py
```

This will compare:
1. CF-ALS with label-aware confidence (α=1.0)
2. CF-PyTorch with exact optimization
3. Baselines (simple average, stacking)

Across 3 imbalance levels (10%, 5%, 1% positive).

---

## Implementation Recommendations

### For Research/Development (Use Both)

**Phase 1 - Fast iteration with ALS:**
```python
# Quick experiments
trainer_als = CFEnsembleTrainer(
    latent_dim=20,
    use_label_aware_confidence=True,
    max_iter=100
)
trainer_als.fit(data)  # Fast prototyping
```

**Phase 2 - Optimize with PyTorch:**
```python
# Best performance
trainer_pt = CFEnsemblePyTorchTrainer(
    latent_dim=20,
    max_epochs=200,
    optimizer='adam'
)
trainer_pt.fit(data)  # Production quality
```

### For Production (Choose One)

**CPU-constrained → ALS:**
```python
# Optimized for speed
trainer = CFEnsembleTrainer(
    latent_dim=20,
    use_label_aware_confidence=True,
    label_aware_alpha=1.0,
    lambda_reg=0.01
)
```

**GPU-available → PyTorch:**
```python
# Optimized for accuracy
trainer = CFEnsemblePyTorchTrainer(
    latent_dim=20,
    device='cuda',
    max_epochs=200,
    optimizer='adamw'
)
```

---

## Conclusion

**Key Takeaways:**

1. **ALS cannot exactly optimize the combined loss** (supervision term is non-quadratic)
2. **Label-aware confidence is a smart approximation** (incorporates supervision via weighting)
3. **PyTorch is the exact solution** (like KD, VAE, all modern ML)
4. **Both approaches are valid** (trade speed vs. accuracy)

**Recommendation:**
- **Research:** Use PyTorch (exact, flexible, extensible)
- **Production:** Evaluate both, choose based on constraints
- **Default:** Start with PyTorch unless strong reason to use ALS

The failure wasn't using ALS—it was using ALS **without label-aware confidence**. With label-aware confidence, ALS becomes a reasonable fast approximation!

---

## References

1. **Matrix Factorization:** Koren et al. (2009) - ALS for recommender systems
2. **Knowledge Distillation:** Hinton et al. (2015) - Combined soft/hard targets
3. **VAE:** Kingma & Welling (2014) - Reparameterization trick
4. **Multi-task Learning:** Chen et al. (2018) - Joint gradient descent
