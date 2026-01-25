# Failure Mode: Aggregator Weight Collapse

**Status:** 🚨 Active Bug  
**Severity:** Critical (destroys predictions)  
**Discovered:** 2026-01-25  
**Fix Status:** Under investigation

---

## TL;DR

During alternating optimization (ALS updates X,Y → aggregator updates θ), the **aggregator weights collapse to near-zero**, bias dominates, and all predictions become constant. This completely destroys model performance.

```
Reconstruction: EXCELLENT ✅ (RMSE = 0.058)
Aggregator weights: [0.009, -0.013, 0.008, ...] ❌ (near zero!)
Final predictions: ALL ≈ 0.398 ❌ (constant!)
```

---

## Symptoms

### How to Detect

1. **Predictions have no variance:**
   ```python
   predictions = trainer.predict()
   print(f"Std: {np.std(predictions)}")  # < 0.001
   ```

2. **Aggregator weights near zero:**
   ```python
   w = trainer.aggregator.get_weights()
   print(f"Weights: {w}")  # All ~ 0.01
   print(f"Sum: {np.sum(w)}")  # ~ 0.01 instead of 1.0
   ```

3. **Bias dominates predictions:**
   ```python
   b = trainer.aggregator.b
   print(f"Bias: {b}")  # e.g., -0.414
   print(f"sigmoid(bias): {1/(1+np.exp(-b))}")  # ≈ 0.398
   # All predictions ≈ sigmoid(bias)
   ```

4. **Performance catastrophically bad:**
   ```
   Simple Average: 1.000 PR-AUC ✅
   CF-Ensemble:    0.056 PR-AUC ❌ (95% worse!)
   ```

---

## Root Cause Analysis

### The Alternating Optimization Problem

**Training loop structure:**
```python
for iteration in range(max_iter):
    # 1. Update X (fix Y) via ALS
    X = update_classifier_factors(Y, R, C, λ)
    
    # 2. Update Y (fix X) via ALS
    Y = update_instance_factors(X, R, C, λ)
    
    # 3. Update aggregator θ (fix X, Y) via gradient descent
    R_hat = X.T @ Y
    aggregator.update(R_hat, labels, lr)
```

**The problem:** Aggregator learns on a **moving target** (R_hat changes every iteration).

### Why Weights Collapse

**Hypothesis 1: Conflicting Objectives**

ALS minimizes:
$$\mathcal{L}_{\text{ALS}} = \sum c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2)$$

Aggregator minimizes:
$$\mathcal{L}_{\text{agg}} = \sum_{i \in \mathcal{L}} CE(y_i, \sigma(w^\top \hat{r}_i + b))$$

These are optimized **separately** with **conflicting** gradients:
- ALS tries to make R_hat ≈ R (preserve diversity)
- Aggregator tries to make predictions match labels (collapse to one value if imbalanced)

**Result:** Weights get pushed toward zero by gradient updates.

**Hypothesis 2: Non-Stationary Target**

```python
# Iteration 1:
R_hat_1 = X_1.T @ Y_1
aggregator.fit(R_hat_1)  # Learn weights for R_hat_1

# Iteration 2:
X_2, Y_2 = als_update(...)  # R_hat changes!
R_hat_2 = X_2.T @ Y_2  # Different from R_hat_1
aggregator.update(R_hat_2)  # Previously learned weights may be wrong
```

Aggregator is "chasing" a moving target, never converging.

**Hypothesis 3: Imbalanced Gradient Magnitudes**

```python
# ALS updates (large changes)
X_new = (YCY^T + λI)^(-1) YCr  # Matrix inversion, can be large

# Aggregator updates (small gradients with typical LR)
w -= lr * ∇_w CE  # lr=0.1, gradients ~ 0.01
```

If ALS makes large changes to R_hat, aggregator gradients become unreliable.

**Hypothesis 4: Class Imbalance Dominates Gradients** ⭐ **CONFIRMED!**

With 10% positive rate, the gradient formula is biased:
```python
residual = y_pred - y_true
# Positives (10%): residual ≈ -0.32 (negative, trying to increase pred)
# Negatives (90%): residual ≈ +0.56 (positive, trying to decrease pred)

grad_w = (R_hat @ residual) / len(residual)
# Averages equally over all instances
# But negatives (90%) DOMINATE the sum!
# Result: grad_w ≈ +0.09 (consistently POSITIVE)

w -= lr * grad_w
# w -= 0.1 * 0.09 = w decreases by 0.009 each iteration
# After 100 iterations: w → 0 (collapsed!)
```

**Mathematical proof from diagnostic:**
```
Iteration 0: weights=[0.20, 0.20, ...], grad_w=[+0.09, +0.09, ...]
Iteration 1: weights=[0.19, 0.19, ...], grad_w=[+0.09, +0.09, ...]
...
Iteration 20: weights=[0.02, 0.02, ...], grad_w=[+0.04, +0.04, ...]
```

Gradients remain **consistently positive** because the 90% negative class dominates the average!

### Diagnostic Evidence

**From deep diagnostic trace:**
```
Iteration 0:
  Weights: [0.20, 0.20, 0.20, 0.20, 0.20]  # Uniform init
  Bias: 0.0
  Predictions: varied

Iteration 10:
  Weights: [0.009, -0.013, 0.008, 0.003, 0.008]  # Collapsed!
  Bias: -0.414
  Predictions: ALL ≈ 0.398 (constant)

Reconstruction quality: RMSE = 0.058 (excellent!)
R_hat PR-AUC: 0.966 (excellent!)
```

**Conclusion:** The problem is NOT reconstruction. It's the aggregator learning dynamics.

---

## Why This Doesn't Happen in PyTorch

**PyTorch joint optimization:**
```python
loss = ρ * recon_loss + (1-ρ) * sup_loss
loss.backward()  # Unified gradients!

# All parameters updated TOGETHER with respect to SAME objective
X -= lr * ∇_X loss
Y -= lr * ∇_Y loss
θ -= lr * ∇_θ loss
```

**Key differences:**
1. **Single unified objective** (not alternating)
2. **Consistent gradients** (all w.r.t. same loss)
3. **No moving target** (all parameters updated simultaneously)

**Prediction:** PyTorch should NOT have this issue.

---

## Example: When It Occurs

### Scenario 1: Imbalanced Data with Good Base Models

```python
# Generate excellent base models (0.75 PR-AUC)
R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
    positive_rate=0.10,  # 10% minority
    target_quality=0.70,
    random_state=42
)

# Simple average works perfectly
simple_avg = np.mean(R, axis=0)
# PR-AUC: 1.000 ✅

# CF-Ensemble collapses
trainer = CFEnsembleTrainer(latent_dim=20, rho=0.5)
trainer.fit(data)
predictions = trainer.predict()
# PR-AUC: 0.056 ❌ (all predictions ≈ 0.398)
```

**Why:** With imbalanced data, majority class dominates gradients. Weights get pushed to minimize loss on majority (zeros), which means w → 0.

### Scenario 2: Small Learning Rate Makes It Worse

```python
# With aggregator_lr = 0.01 (very small)
# Weights decay very slowly but consistently
# Eventually collapse after 100+ iterations

# With aggregator_lr = 1.0 (large)
# Weights may oscillate but less likely to collapse completely
```

### Scenario 3: High ρ (More Reconstruction Focus)

```python
# With ρ = 0.9 (mostly reconstruction)
# Supervised updates are weak
# Aggregator doesn't learn much, weights stay near init

# With ρ = 0.5 (balanced)
# Supervised updates compete with reconstruction
# More likely to cause instability
```

---

## Proposed Fixes

### Fix 1: Freeze Aggregator Initially ⏸️

**Strategy:** Let X, Y stabilize before enabling aggregator learning.

```python
class CFEnsembleTrainer:
    def __init__(self, ..., freeze_aggregator_iters=50):
        self.freeze_aggregator_iters = freeze_aggregator_iters
    
    def fit(self, data):
        for iteration in range(self.max_iter):
            # Update X, Y
            self.X = update_classifier_factors(...)
            self.Y = update_instance_factors(...)
            
            # Only update aggregator after warmup
            if iteration >= self.freeze_aggregator_iters:
                self.aggregator.update(...)
```

**Pros:**
- Simple to implement
- Lets reconstruction stabilize first
- Should reduce moving target problem

**Cons:**
- ⚠️ Very empirical (how many iterations?)
- Different datasets may need different freeze periods
- Doesn't address root cause

**Expected outcome:** Weights may not collapse if R_hat is stable.

### Fix 2: Weight Regularization 📏

**Strategy:** Add L2 penalty to keep weights from going to zero.

```python
# In aggregator update
grad_w = (R_hat @ residual) / len(residual)
grad_w += λ_w * self.w  # L2 regularization

# Or: constraint to keep |w| > min_value
self.w = np.maximum(np.abs(self.w), 0.01) * np.sign(self.w)
```

**Pros:**
- Prevents complete collapse
- Well-established technique
- Easy to implement

**Cons:**
- Adds another hyperparameter (λ_w)
- May not fix underlying instability
- Could prevent learning optimal weights

### Fix 3: Momentum / Adaptive Learning Rate 📈

**Strategy:** Use Adam-like updates for aggregator.

```python
# Track exponential moving average of gradients
self.m_w = β * self.m_w + (1-β) * grad_w
self.w -= lr * self.m_w
```

**Pros:**
- Smooths out noisy gradients
- Standard in deep learning
- May stabilize updates

**Cons:**
- More complex
- Still doesn't fix moving target
- Adds hyperparameters (β, etc.)

### Fix 4: Periodic Re-initialization 🔄

**Strategy:** If weights collapse, reset to uniform.

```python
if np.std(self.w) < 0.01:  # Collapsed
    print("Resetting aggregator weights...")
    self.w = np.ones(m) / m
    self.b = 0.0
```

**Pros:**
- Escapes bad local minimum
- Simple heuristic
- May help in practice

**Cons:**
- Hacky solution
- May reset when legitimately learned
- Doesn't fix root cause

### Fix 5: Use Mean Aggregator (Skip Learning) 🔧

**Strategy:** Don't learn weights at all.

```python
trainer = CFEnsembleTrainer(
    aggregator_type='mean',  # No learnable parameters
    ...
)
```

**Pros:**
- Eliminates the problem entirely
- Proves reconstruction works
- Simplest solution

**Cons:**
- ❌ Defeats the purpose of learning
- Can't leverage classifier strengths
- Not a real solution

**Use case:** Debugging/validation only.

### Fix 6: Switch to PyTorch 🔥

**Strategy:** Use joint optimization instead of alternating.

```python
trainer = CFEnsemblePyTorchTrainer(
    latent_dim=20,
    rho=0.5,
    max_epochs=200,
    optimizer='adam'
)
```

**Pros:**
- ✅ Should avoid alternating optimization issues
- ✅ Unified gradients
- ✅ Well-tested approach (like KD, VAE)

**Cons:**
- Requires PyTorch
- Slower on CPU
- Different hyperparameters to tune

**Expected outcome:** Should work correctly.

---

## Recommended Solution

### Short-term: Fix 1 (Freeze Aggregator)

**Implementation:**
1. Add `freeze_aggregator_iters` parameter
2. Skip aggregator updates for first N iterations
3. Test on benchmark data
4. Tune N empirically (try 20, 50, 100)

**Why:** Simple, likely to help, easy to test.

### Medium-term: Fix 6 (PyTorch)

**Implementation:**
1. We already have `CFEnsemblePyTorchTrainer`
2. Test it on same data
3. Compare results with ALS + freeze

**Why:** Theoretically correct, avoids alternating optimization issues.

### Long-term: Redesign Alternating Optimization

**Options:**
1. **Coordinate descent on full loss:** Update one parameter at a time w.r.t. full L_CF
2. **Block coordinate descent:** Update X, Y together, then θ
3. **Hybrid:** Few ALS steps → Few aggregator steps → Repeat

**Why:** Addresses root cause, more principled.

---

## Testing Strategy

### Test 1: Confirm Freeze Fixes It

```python
# Train with freeze
trainer = CFEnsembleTrainer(
    freeze_aggregator_iters=50,  # NEW PARAMETER
    max_iter=200,
    aggregator_lr=0.1
)
trainer.fit(data)

# Check if weights are healthy
w = trainer.aggregator.get_weights()
assert np.std(w) > 0.05, "Weights still collapsed!"
assert np.sum(np.abs(w)) > 0.5, "Weights too small!"

# Check performance
predictions = trainer.predict()
pr_auc = average_precision_score(y_test, predictions[test_idx])
pr_simple = average_precision_score(y_test, np.mean(R_test, axis=0))

assert pr_auc > pr_simple * 0.9, "Still worse than simple average!"
```

### Test 2: PyTorch Comparison

```python
# Train with PyTorch
trainer_pt = CFEnsemblePyTorchTrainer(
    latent_dim=20,
    rho=0.5,
    max_epochs=200
)
trainer_pt.fit(data)

# Check if it avoids the issue
predictions_pt = trainer_pt.predict()
pr_pt = average_precision_score(y_test, predictions_pt[test_idx])

print(f"ALS+freeze: {pr_auc:.3f}")
print(f"PyTorch:    {pr_pt:.3f}")
print(f"Simple avg: {pr_simple:.3f}")
```

### Test 3: Convergence Analysis

```python
# Track weights over time
weight_history = []
for iteration in range(max_iter):
    # ... training ...
    weight_history.append(trainer.aggregator.w.copy())

# Plot
plt.plot(weight_history)
plt.xlabel("Iteration")
plt.ylabel("Weight value")
plt.title("Aggregator Weight Evolution")
plt.show()

# Should see:
# - Frozen phase: weights constant
# - Learning phase: weights change but don't collapse
```

---

## Related Issues

### Similar Problems in Literature

1. **Alternating optimization instability:**
   - EM algorithm can oscillate
   - Solution: Damping, momentum, early stopping

2. **Non-convex optimization:**
   - Local minima, saddle points
   - Solution: Random restarts, better initialization

3. **Vanishing gradients:**
   - Common in deep learning
   - Solution: Better activation, normalization, gradient clipping

### Analogous to CF-Ensemble

Our problem is a combination:
- **Alternating** (like EM)
- **Non-convex** (like deep learning)
- **Imbalanced data** (vanishing gradients on minority)

**Key difference from standard EM:** We're not doing E-step/M-step on same objective. We're optimizing **different objectives** (recon vs. supervised) in alternating fashion.

---

## Prevention

### Design Principles to Avoid This

1. **Unified objectives:** Optimize all parameters w.r.t. same loss (PyTorch approach)
2. **Gradual learning:** Introduce supervision slowly (curriculum learning)
3. **Stabilization:** Use techniques like batch normalization, gradient clipping
4. **Monitoring:** Track weight norms, gradients, detect collapse early

### Warning Signs During Training

```python
# Add to training loop
if iteration % 10 == 0:
    w_norm = np.linalg.norm(trainer.aggregator.w)
    if w_norm < 0.1:
        warnings.warn(f"Aggregator weights collapsing! Norm={w_norm:.4f}")
```

---

## References

1. **Alternating Least Squares:** Hu et al. (2008) - Shows ALS works for single objective
2. **EM Algorithm Instability:** Dempster et al. (1977) - Classic EM issues
3. **Multi-task Learning:** Chen et al. (2018) "GradNorm" - Balancing multiple losses
4. **Vanishing Gradients:** Bengio et al. (1994) - Gradient flow problems

---

## Status & Next Steps

**Current Status:** 🚨 Bug documented, fixes proposed, testing in progress

**Immediate Actions:**
1. ✅ Document failure mode
2. 🔲 Implement Fix 1 (freeze aggregator)
3. 🔲 Test on benchmark data
4. 🔲 Compare with PyTorch
5. 🔲 Choose best solution

**Success Criteria:**
- ✅ CF-Ensemble PR-AUC > Simple Average
- ✅ Weights remain healthy (std > 0.05)
- ✅ Predictions have variance (std > 0.1)
- ✅ Converges reliably across random seeds

---

**Last Updated:** 2026-01-25  
**Next Review:** After testing freeze fix
