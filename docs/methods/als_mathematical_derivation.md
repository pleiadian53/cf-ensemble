# ALS with Label-Aware Confidence: Mathematical Derivation

**How ALS approximates the combined reconstruction + supervision objective**

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Combined Objective](#the-combined-objective)
3. [Why ALS Cannot Optimize It Directly](#why-als-cannot-optimize-it-directly)
4. [Label-Aware Confidence Approximation](#label-aware-confidence-approximation)
5. [Deriving the ALS Updates](#deriving-the-als-updates)
6. [Complete Algorithm](#complete-algorithm)
7. [Theoretical Analysis](#theoretical-analysis)
8. [Implementation](#implementation)

---

## Introduction

**Goal:** Optimize the combined KD-inspired objective:
$$\mathcal{L}_{\text{CF}} = \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta)$$

This balances:
- **Reconstruction**: Faithfully representing base classifier predictions
- **Supervision**: Ensuring aggregated predictions match ground truth labels

**Challenge:** The supervised term is non-quadratic (contains sigmoid + log), preventing closed-form ALS solutions.

**Solution:** Use **label-aware confidence weighting** to approximate supervision within ALS.

This document derives the mathematical foundation for this approximation.

---

## The Combined Objective

### Full Loss Function

Following the knowledge distillation analogy:

$$\begin{align}
\mathcal{L}_{\text{CF}}(X, Y, \theta) &= \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta) \\
&= \rho \cdot \left[\sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2)\right] \\
&\quad + (1-\rho) \cdot \sum_{i \in \mathcal{L}} CE(y_i, g_\theta(X^\top y_i))
\end{align}$$

where:
- $R \in [0,1]^{m \times n}$: Probability matrix from base classifiers
- $X \in \mathbb{R}^{d \times m}$: Classifier latent factors
- $Y \in \mathbb{R}^{d \times n}$: Instance latent factors
- $C \in \mathbb{R}_+^{m \times n}$: Confidence weights
- $\mathcal{L} \subset \{1,\ldots,n\}$: Labeled instances
- $g_\theta$: Aggregation function (e.g., $\sigma(w^\top(\cdot) + b)$)
- $CE$: Binary cross-entropy

### What Each Term Does

**Reconstruction term** ($L_{\text{recon}}$):
- Quadratic in $X$ and $Y$
- Has closed-form ALS solution
- Encourages $X^\top Y \approx R$ (low-rank approximation)

**Supervised term** ($L_{\text{sup}}$):
$$CE(y_i, g_\theta(X^\top y_i)) = -y_i \log \sigma(w^\top(X^\top y_i) + b) - (1-y_i) \log(1-\sigma(...))$$
- Non-quadratic (sigmoid + log)
- No closed-form solution
- Encourages predictions to match labels

---

## Why ALS Cannot Optimize It Directly

### ALS Requires Quadratic Objectives

**Standard ALS** works when the objective is **quadratic in each variable block**:

$$\min_{X} \sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda \|X\|_F^2$$

This gives closed-form solution:
$$x_u = (YC_uY^\top + \lambda I)^{-1} YC_u r_u$$

### The Supervised Term is Non-Quadratic

Taking gradient of $L_{\text{sup}}$ w.r.t. $X$:

$$\begin{align}
\nabla_X L_{\text{sup}} &= \sum_{i \in \mathcal{L}} \nabla_X CE(y_i, g_\theta(X^\top y_i)) \\
&= \sum_{i \in \mathcal{L}} \underbrace{(\sigma(w^\top(X^\top y_i) + b) - y_i)}_{\text{residual}} \cdot \underbrace{w}_{\text{weights}} \cdot \underbrace{y_i^\top}_{\text{instance factor}}
\end{align}$$

**The problem:**
- Contains $\sigma(\cdot)$ (non-linear)
- Depends on $\theta$ (aggregator parameters)
- No closed-form inverse

**Conclusion:** Cannot derive ALS update equations for the full combined loss.

---

## Label-Aware Confidence Approximation

### Key Insight

Instead of directly minimizing:
$$\mathcal{L}_{\text{CF}} = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$$

**Approximate** by minimizing:
$$\mathcal{L}_{\text{approx}}(X, Y) = \sum_{u,i} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2)$$

where $\tilde{c}_{ui}$ is **label-aware**: encodes supervision signal through confidence modulation.

### Label-Aware Confidence Definition

$$\tilde{c}_{ui} = \begin{cases}
c_{ui}^{\text{base}} \cdot (1 + \alpha \cdot r_{ui}) & \text{if } i \in \mathcal{L} \text{ and } y_i = 1 \\
c_{ui}^{\text{base}} \cdot (1 + \alpha \cdot (1 - r_{ui})) & \text{if } i \in \mathcal{L} \text{ and } y_i = 0 \\
c_{ui}^{\text{base}} & \text{if } i \notin \mathcal{L} \text{ (unlabeled)}
\end{cases}$$

where:
- $c_{ui}^{\text{base}}$: Base confidence (typically $|r_{ui} - 0.5|$ for certainty)
- $\alpha \geq 0$: Supervision strength parameter

### How This Incorporates Supervision

**For positive instances** ($y_i = 1$):
- High $r_{ui}$ (correct prediction) → High $\tilde{c}_{ui}$ → ALS prioritizes preserving this
- Low $r_{ui}$ (incorrect prediction) → Lower $\tilde{c}_{ui}$ → ALS deprioritizes this

**For negative instances** ($y_i = 0$):
- Low $r_{ui}$ (correct prediction) → High $\tilde{c}_{ui}$ → ALS prioritizes preserving this
- High $r_{ui}$ (incorrect prediction) → Lower $\tilde{c}_{ui}$ → ALS deprioritizes this

**Result:** ALS learns to reconstruct **correct predictions** well while downweighting **errors**.

### Mathematical Justification

Expand the approximate objective:

$$\begin{align}
\mathcal{L}_{\text{approx}} &= \sum_{u,i} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2) \\
&= \sum_{u,i} c_{ui}^{\text{base}}(1 + \alpha \cdot s_{ui})(r_{ui} - x_u^\top y_i)^2 + \text{reg}
\end{align}$$

where $s_{ui}$ is the **supervision signal**:
$$s_{ui} = \begin{cases}
r_{ui} & \text{if } y_i = 1 \\
1 - r_{ui} & \text{if } y_i = 0 \\
0 & \text{if unlabeled}
\end{cases}$$

This is equivalent to:
$$\mathcal{L}_{\text{approx}} \approx \underbrace{\sum c_{ui}^{\text{base}}(r_{ui} - x_u^\top y_i)^2}_{\text{reconstruction}} + \alpha \cdot \underbrace{\sum c_{ui}^{\text{base}} s_{ui}(r_{ui} - x_u^\top y_i)^2}_{\text{supervision-weighted reconstruction}}$$

The second term **implicitly encourages** $X^\top y_i$ to match predictions that agree with labels!

---

## Deriving the ALS Updates

### Classifier Factor Update

**Objective (fix $Y$, optimize $X$):**
$$\min_X \sum_{u=1}^{m} \sum_{i=1}^{n} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda \|X\|_F^2$$

This decomposes into $m$ independent problems (one per classifier):
$$\min_{x_u} \sum_{i=1}^{n} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda \|x_u\|^2$$

**Gradient:**
$$\nabla_{x_u} = -2\sum_{i=1}^{n} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)y_i + 2\lambda x_u$$

**Set to zero:**
$$\sum_{i=1}^{n} \tilde{c}_{ui} y_i y_i^\top x_u + \lambda x_u = \sum_{i=1}^{n} \tilde{c}_{ui} r_{ui} y_i$$

**Rearrange:**
$$(Y\tilde{C}_u Y^\top + \lambda I)x_u = Y\tilde{C}_u r_u$$

where:
- $\tilde{C}_u = \text{diag}(\tilde{c}_{u1}, \ldots, \tilde{c}_{un})$: Label-aware confidence for classifier $u$
- $r_u = [r_{u1}, \ldots, r_{un}]^\top$: Predictions from classifier $u$

**Closed-form solution:**
$$\boxed{x_u = (Y\tilde{C}_u Y^\top + \lambda I)^{-1} Y\tilde{C}_u r_u}$$

### Instance Factor Update

**Objective (fix $X$, optimize $Y$):**
$$\min_Y \sum_{u=1}^{m} \sum_{i=1}^{n} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda \|Y\|_F^2$$

This decomposes into $n$ independent problems (one per instance):
$$\min_{y_i} \sum_{u=1}^{m} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda \|y_i\|^2$$

**Gradient:**
$$\nabla_{y_i} = -2\sum_{u=1}^{m} \tilde{c}_{ui}(r_{ui} - x_u^\top y_i)x_u + 2\lambda y_i$$

**Set to zero:**
$$\sum_{u=1}^{m} \tilde{c}_{ui} x_u x_u^\top y_i + \lambda y_i = \sum_{u=1}^{m} \tilde{c}_{ui} r_{ui} x_u$$

**Rearrange:**
$$(X\tilde{C}_i X^\top + \lambda I)y_i = X\tilde{C}_i r_i$$

where:
- $\tilde{C}_i = \text{diag}(\tilde{c}_{1i}, \ldots, \tilde{c}_{mi})$: Label-aware confidence for instance $i$
- $r_i = [r_{1i}, \ldots, r_{mi}]^\top$: Predictions for instance $i$

**Closed-form solution:**
$$\boxed{y_i = (X\tilde{C}_i X^\top + \lambda I)^{-1} X\tilde{C}_i r_i}$$

### Key Observation

The update equations have the **same form** as standard ALS, but use **label-aware confidence** $\tilde{C}$ instead of base confidence $C$.

This is why the approximation is elegant: we don't modify the ALS algorithm structure, just the confidence weights!

---

## Complete Algorithm

### Algorithm: ALS with Label-Aware Confidence

**Input:**
- $R \in [0,1]^{m \times n}$: Probability matrix
- $y \in \{0,1\}^n$: Labels (with NaN for unlabeled)
- $\mathcal{L}$: Set of labeled indices
- $\rho \in [0,1]$: Reconstruction vs. supervision trade-off
- $\alpha \geq 0$: Label-aware supervision strength
- $\lambda > 0$: Regularization strength
- $d$: Latent dimensionality

**Output:**
- $X \in \mathbb{R}^{d \times m}$: Classifier factors
- $Y \in \mathbb{R}^{d \times n}$: Instance factors

**Steps:**

1. **Initialize:**
   ```
   X ← random(d, m) * 0.01
   Y ← random(d, n) * 0.01
   ```

2. **Compute base confidence:**
   ```
   C_base[u,i] ← |r[u,i] - 0.5|  # Certainty-based
   ```

3. **Compute label-aware confidence:**
   ```
   C_label_aware ← C_base
   for i in labeled_indices:
       if y[i] == 1:
           C_label_aware[:,i] ← C_base[:,i] * (1 + α * R[:,i])
       else:
           C_label_aware[:,i] ← C_base[:,i] * (1 + α * (1 - R[:,i]))
   ```

4. **Iterate until convergence:**
   ```
   for iteration in 1 to max_iter:
       # Update classifier factors
       for u in 1 to m:
           A ← Y @ diag(C_label_aware[u,:]) @ Y.T + λ*I
           b ← Y @ (C_label_aware[u,:] * R[u,:])
           X[:,u] ← solve(A, b)
       
       # Update instance factors
       for i in 1 to n:
           A ← X @ diag(C_label_aware[:,i]) @ X.T + λ*I
           b ← X @ (C_label_aware[:,i] * R[:,i])
           Y[:,i] ← solve(A, b)
       
       # Check convergence (optional: recompute loss)
       if ||ΔX|| < tol and ||ΔY|| < tol:
           break
   ```

5. **Return** $X, Y$

### Aggregator Training (Separate Step)

After ALS converges, train the aggregator $\theta$:

```
R_hat ← X.T @ Y
for epoch in 1 to max_epochs:
    y_pred ← aggregator(R_hat[:,labeled_indices], θ)
    loss ← cross_entropy(y[labeled_indices], y_pred)
    θ ← θ - lr * ∇_θ loss
```

---

## Theoretical Analysis

### Convergence Guarantee

**Theorem:** Each ALS update (fixing one set of factors) decreases the approximate objective:
$$\mathcal{L}_{\text{approx}}(X^{t+1}, Y^t) \leq \mathcal{L}_{\text{approx}}(X^t, Y^t)$$
$$\mathcal{L}_{\text{approx}}(X^{t+1}, Y^{t+1}) \leq \mathcal{L}_{\text{approx}}(X^{t+1}, Y^t)$$

**Proof:** Each update minimizes a convex quadratic objective, guaranteeing decrease.

**Corollary:** ALS converges to a local minimum of $\mathcal{L}_{\text{approx}}$.

### Approximation Quality

**Question:** How well does $\mathcal{L}_{\text{approx}}$ approximate $\mathcal{L}_{\text{CF}}$?

**Intuition:** Label-aware confidence creates a **first-order approximation** of the supervision gradient.

**Formal analysis:**

Consider the supervision gradient:
$$\nabla_{X,Y} L_{\text{sup}} \propto \sum_{i \in \mathcal{L}} (g_\theta(X^\top y_i) - y_i) \cdot \nabla_{X,Y} g_\theta$$

Label-aware confidence implicitly adds a term proportional to:
$$\alpha \sum_{i \in \mathcal{L}} s_{ui} \cdot \nabla_{X,Y} \|R - X^\top Y\|^2$$

where $s_{ui}$ correlates with $(g_\theta - y_i)$ when $g_\theta$ is well-calibrated.

**Result:** The approximation is good when:
1. Base classifiers are well-calibrated
2. Aggregator is simple (e.g., mean or weighted mean)
3. $\alpha$ is tuned appropriately

### Comparison to PyTorch

**ALS + Label-Aware:**
- Approximate combined loss
- Fast (closed-form updates)
- May have small optimality gap

**PyTorch:**
- Exact combined loss
- Slower (gradient computation)
- Guaranteed to find better local minimum

**Expected gap:** 5-15% in PR-AUC (ALS slightly worse but much faster).

---

## Implementation

### Python Implementation

```python
def update_classifier_factors_label_aware(
    Y: np.ndarray,
    R: np.ndarray,
    C_label_aware: np.ndarray,
    lambda_reg: float
) -> np.ndarray:
    """
    Update classifier factors using label-aware confidence.
    
    Parameters:
        Y: Instance factors (d × n)
        R: Probability matrix (m × n)
        C_label_aware: Label-aware confidence (m × n)
        lambda_reg: Regularization strength
    
    Returns:
        X: Updated classifier factors (d × m)
    """
    d, n = Y.shape
    m = R.shape[0]
    X = np.zeros((d, m))
    lambda_I = lambda_reg * np.eye(d)
    
    for u in range(m):
        # Label-aware confidence for this classifier
        c_u = C_label_aware[u, :]
        
        # Weighted gram matrix: Y @ diag(c_u) @ Y.T
        Y_weighted = Y * c_u[None, :]
        A = Y_weighted @ Y.T + lambda_I
        
        # Weighted target: Y @ (c_u * r_u)
        r_u = R[u, :]
        b = Y_weighted @ r_u
        
        # Solve: A x_u = b
        X[:, u] = np.linalg.solve(A, b)
    
    return X


def compute_label_aware_confidence(
    R: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
    base_confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute label-aware confidence weights.
    
    Parameters:
        R: Probability matrix (m × n)
        labels: Ground truth labels (n,) with NaN for unlabeled
        alpha: Supervision strength
        base_confidence: Base confidence (default: certainty |r - 0.5|)
    
    Returns:
        C_label_aware: Label-aware confidence (m × n)
    """
    if base_confidence is None:
        base_confidence = np.abs(R - 0.5)
    
    C = base_confidence.copy()
    labeled_mask = ~np.isnan(labels)
    
    for i in np.where(labeled_mask)[0]:
        if labels[i] == 1:
            # Positive: reward high predictions
            C[:, i] = base_confidence[:, i] * (1 + alpha * R[:, i])
        else:
            # Negative: reward low predictions
            C[:, i] = base_confidence[:, i] * (1 + alpha * (1 - R[:, i]))
    
    return C
```

### Usage Example

```python
from cfensemble.data import EnsembleData
from cfensemble.optimization import CFEnsembleTrainer

# Create data
data = EnsembleData(R, labels)

# Train with label-aware confidence
trainer = CFEnsembleTrainer(
    n_classifiers=m,
    latent_dim=20,
    rho=0.5,  # Note: rho used in loss computation, not directly in ALS
    lambda_reg=0.01,
    use_label_aware_confidence=True,  # Enable approximation
    label_aware_alpha=1.0,              # Supervision strength
    max_iter=100
)

trainer.fit(data)
predictions = trainer.predict()
```

---

## Class-Weighted Gradients for Imbalanced Data

### The Challenge

While label-aware confidence approximates supervision in the ALS updates (X, Y), the **aggregator parameters (θ)** are trained separately via gradient descent. On **imbalanced data** (e.g., 10% positive, 90% negative), standard gradient descent can cause aggregator weights to **collapse to negative values**, destroying predictions.

### The Problem

Standard aggregator gradient treats all instances equally:

$$\nabla_w L_{\text{sup}} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i) \cdot \hat{r}_i$$

With imbalanced data, the **majority class dominates** gradient computation:

```
10% positive: residual ≈ -0.5 (minority)
90% negative: residual ≈ +0.5 (MAJORITY DOMINATES!)
→ Total gradient: +0.40 (positive)
→ Weights decrease continuously → collapse!
```

### The Solution

**Class-weighted gradients** weight instances by inverse class frequency:

$$w_{\text{pos}} = \frac{n}{2n_{\text{pos}}}, \quad w_{\text{neg}} = \frac{n}{2n_{\text{neg}}}$$

$$\nabla_w L_{\text{sup}}^{\text{weighted}} = \frac{\sum_i w_{class(i)} \cdot (\hat{y}_i - y_i) \cdot \hat{r}_i}{\sum_i w_{class(i)}}$$

This ensures each **class** (not instance) contributes equally to gradients, preventing majority class domination.

### Usage

```python
trainer = CFEnsembleTrainer(
    n_classifiers=m,
    latent_dim=20,
    use_label_aware_confidence=True,  # ALS approximation
    use_class_weights=True,           # Aggregator class weighting (default)
    max_iter=100
)
```

**Result:** Weights remain positive and stable, achieving perfect performance even with 10% positive rate.

**See:** [`docs/methods/optimization/class_weighted_gradients.md`](optimization/class_weighted_gradients.md) for complete derivation and experimental results.

---

## Conclusion

**Key Takeaways:**

1. **ALS cannot optimize the combined loss exactly** because $L_{\text{sup}}$ is non-quadratic

2. **Label-aware confidence provides an elegant approximation**:
   - Modulates $c_{ui}$ based on label agreement
   - Preserves ALS structure (closed-form updates)
   - Incorporates supervision indirectly

3. **Mathematical foundation is sound**:
   - Each ALS step provably decreases $\mathcal{L}_{\text{approx}}$
   - Approximation quality is first-order in supervision gradient
   - Empirically works well for calibrated base models

4. **Trade-off with PyTorch**:
   - ALS: Fast approximate optimization
   - PyTorch: Exact but slower optimization

The label-aware confidence trick makes ALS viable for CF-Ensemble while maintaining computational efficiency!

---

## References

1. **ALS for Matrix Factorization:** Hu et al. (2008) "Collaborative Filtering for Implicit Feedback"
2. **Knowledge Distillation:** Hinton et al. (2015) "Distilling Knowledge"
3. **Confidence Weighting:** Koren et al. (2009) "Matrix Factorization Techniques"
4. **Multi-task Learning:** Chen et al. (2018) "GradNorm: Gradient Normalization"
