# Optimization Techniques Summary

**Quick Reference:** Where do different techniques apply in CF-Ensemble?

---

## The Three Techniques

### 1. Label-Aware Confidence ⚙️
- **Purpose:** Approximate supervision in closed-form ALS
- **Method:** Modulates confidence matrix C based on label agreement
- **Parameters:** X, Y (latent factors)
- **Applies to:** ALS only

### 2. Class-Weighted Gradients 📊
- **Purpose:** Balance class contributions in gradient descent
- **Method:** Weight instances by inverse class frequency
- **Parameters:** w, b (aggregator) in ALS; all parameters in PyTorch
- **Applies to:** Wherever we use gradient descent

### 3. Focal Loss 🎯
- **Purpose:** Focus on hard examples, down-weight easy ones
- **Method:** Weight instances by $(1-p_t)^\gamma$
- **Parameters:** w, b (aggregator) in ALS; all parameters in PyTorch
- **Applies to:** Wherever we use gradient descent

---

## Quick Decision Guide

### "Which parameters should I set?"

```
Using ALS Trainer?
├─ Is your data imbalanced? (e.g., 10% positive)
│  ├─ YES → use_label_aware_confidence=True ✅ (for X, Y)
│  │         use_class_weights=True ✅ (for w, b)
│  └─ NO  → Can use defaults (both are safe to enable)
│
└─ Do you have easy/hard example variance? (high disagreement)
   ├─ YES → focal_gamma=2.0 ✅ (for w, b)
   └─ NO  → focal_gamma=0.0 (default)

Using PyTorch Trainer?
├─ Is your data imbalanced?
│  ├─ YES → use_class_weights=True ✅ (for all parameters)
│  └─ NO  → Can use default (safe to enable)
│
└─ Do you have easy/hard example variance?
   ├─ YES → focal_gamma=2.0 ✅ (for all parameters)
   └─ NO  → focal_gamma=0.0 (default)
```

---

## Optimization Method Comparison

| Component | ALS Method | PyTorch Method |
|-----------|------------|----------------|
| **Latent factors (X, Y)** | Closed-form ALS<br/>*fast, approximate* | Gradient descent<br/>*slow, exact* |
| **Aggregator (w, b)** | Gradient descent<br/>*iterative* | Gradient descent<br/>*iterative* |
| **Supervision for X, Y** | Label-aware confidence<br/>*approximate* | Direct gradients<br/>*exact* |
| **Supervision for w, b** | Direct BCE loss<br/>*exact* | Direct BCE loss<br/>*exact* |

---

## Where Each Technique Applies

### Quick Reference Table

| Technique | ALS: X, Y | ALS: w, b | PyTorch: All |
|-----------|-----------|-----------|--------------|
| **Label-aware confidence** | ✅ Yes | ❌ No | ❌ No |
| **Class-weighted gradients** | ❌ No | ✅ Yes | ✅ Yes |
| **Focal loss** | ❌ No | ✅ Yes | ✅ Yes |

### Why This Pattern?

```python
# ALS: Hybrid approach
for iteration in range(max_iter):
    X = closed_form_solution(Y, R, C_label_aware, λ)  # Uses label-aware conf
    Y = closed_form_solution(X, R, C_label_aware, λ)  # Uses label-aware conf
    w, b = gradient_descent(X, Y, labels, class_weights, focal)  # Uses class + focal

# PyTorch: Unified approach  
for epoch in range(max_epochs):
    loss = reconstruction + supervised_with_class_weights_and_focal
    loss.backward()  # Gradients flow to ALL parameters
    optimizer.step()  # Updates X, Y, w, b together
```

---

## Recommended Configurations

### For Imbalanced Data (e.g., 10% positive)

**ALS (recommended for speed):**
```python
trainer = CFEnsembleTrainer(
    n_classifiers=10,
    latent_dim=20,
    rho=0.5,
    use_label_aware_confidence=True,  # Handle imbalance in X, Y
    use_class_weights=True,           # Handle imbalance in w, b
    focal_gamma=0.0                   # Optional: add if needed
)
```

**PyTorch (recommended for accuracy):**
```python
trainer = CFEnsemblePyTorchTrainer(
    n_classifiers=10,
    latent_dim=20,
    rho=0.5,
    use_class_weights=True,  # Handle imbalance in all parameters
    focal_gamma=0.0          # Optional: add if needed
)
```

### For Imbalanced + High Disagreement

**ALS:**
```python
trainer = CFEnsembleTrainer(
    use_label_aware_confidence=True,  # Imbalance in X, Y
    use_class_weights=True,           # Imbalance in w, b
    focal_gamma=2.0                   # Hard examples in w, b
)
```

**PyTorch:**
```python
trainer = CFEnsemblePyTorchTrainer(
    use_class_weights=True,  # Imbalance everywhere
    focal_gamma=2.0          # Hard examples everywhere
)
```

---

## Common Misconceptions

### ❌ "Class weighting applies to all parameters in ALS"

**Wrong!** Class-weighted gradients only apply to the **aggregator (w, b)** in ALS. The latent factors (X, Y) use closed-form solutions (no gradients).

**Correct:** ALS uses label-aware confidence for X, Y and class weighting for w, b.

### ❌ "Label-aware confidence applies to PyTorch"

**Wrong!** Label-aware confidence is an ALS-specific **approximation** trick. PyTorch has exact gradients and doesn't need it.

**Correct:** PyTorch uses class-weighted loss for all parameters, no approximation needed.

### ❌ "Focal loss applies to latent factors in ALS"

**Wrong!** Focal loss requires gradients. ALS updates X, Y with closed-form solutions (no gradients).

**Correct:** Focal loss only applies to the aggregator (w, b) in ALS, or all parameters in PyTorch.

---

## Technical Deep Dive

### Why Can't We Apply Class Weighting to ALS Updates?

ALS uses **closed-form solutions** that directly compute the optimal X, Y:

$$X^* = \arg\min_X \|C \odot (R - X^TY)\|_F^2 + \lambda\|X\|_F^2$$

This is solved via:
$$X = (YC^TY^T + \lambda I)^{-1}YC^TR^T$$

**There are no gradients here!** It's a direct matrix equation. We can't "weight" the solution because it's already optimal for the given C matrix.

**Instead:** We modulate C itself (via label-aware weighting) to incorporate supervision.

### Why Does PyTorch Apply Weighting Everywhere?

PyTorch uses **gradient descent** for all parameters:

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta L(\theta)$$

where $\theta = \{X, Y, w, b\}$ are all the parameters.

The loss function:
$$L = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}^{\text{weighted}}$$

When we apply class weighting to $L_{\text{sup}}$, the gradients flow back to **all parameters** via backpropagation:
- $\nabla_X L_{\text{sup}}$ ← affected by class weighting
- $\nabla_Y L_{\text{sup}}$ ← affected by class weighting
- $\nabla_w L_{\text{sup}}$ ← affected by class weighting
- $\nabla_b L_{\text{sup}}$ ← affected by class weighting

---

## See Also

- **[Class-Weighted Gradients](class_weighted_gradients.md)** - Full documentation
- **[Focal Loss](focal_loss.md)** - Full documentation
- **[ALS Mathematical Derivation](../als_mathematical_derivation.md)** - Label-aware confidence
- **[ALS vs PyTorch](../als_vs_pytorch.md)** - Detailed comparison

---

**Last Updated:** 2026-01-25  
**Status:** Reference document for technique applicability
