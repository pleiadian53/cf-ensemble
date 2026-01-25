# CF-Ensemble Quick Reference

**Essential equations and concepts at a glance**

---

## Core Idea

Combine matrix reconstruction with supervised learning, inspired by knowledge distillation:

> **Pure reconstruction reproduces errors. Adding supervision teaches what "signal" means.**

---

## The Complete Objective

$$\boxed{\mathcal{L} = \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta)}$$

### Term 1: Reconstruction Loss

$$L_{\text{recon}} = \sum_{u=1}^m \sum_{i=1}^n c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda \left( \|X\|_F^2 + \|Y\|_F^2 \right)$$

**Purpose**: Faithfully reproduce probability matrix (all points)

**Interpretation**: "Learn latent factors that reconstruct base model predictions"

### Term 2: Supervised Loss

$$L_{\text{sup}} = \sum_{i \in \mathcal{L}} \text{CE}\left(y_i, g_\theta(\hat{r}_{\cdot i})\right)$$

where:
$$\text{CE}(y, \hat{p}) = -y \log \hat{p} - (1-y) \log(1-\hat{p})$$

**Purpose**: Predict correct labels (labeled points only)

**Interpretation**: "Ensure aggregated predictions match ground truth"

---

## Key Objects and Notation

| Symbol | Meaning | Typical Size |
|--------|---------|--------------|
| $R \in [0,1]^{m \times n}$ | Probability matrix | 10 × 1000 |
| $r_{ui}$ | Classifier $u$'s prob for point $i$ | [0, 1] |
| $X \in \mathbb{R}^{d \times m}$ | Classifier latent factors | 20 × 10 |
| $Y \in \mathbb{R}^{d \times n}$ | Instance latent factors | 20 × 1000 |
| $x_u \in \mathbb{R}^d$ | Latent vector for classifier $u$ | dim 20 |
| $y_i \in \mathbb{R}^d$ | Latent vector for point $i$ | dim 20 |
| $\hat{r}_{ui} = x_u^\top y_i$ | Reconstructed probability | [0, 1] |
| $C \in \mathbb{R}_+^{m \times n}$ | Confidence/reliability weights | 10 × 1000 |
| $c_{ui}$ | Trust in $r_{ui}$ | ≥ 0 |
| $\mathcal{L} \subset \{1,\ldots,n\}$ | Labeled point indices | e.g., {1,...,500} |
| $\mathcal{U} \subset \{1,\ldots,n\}$ | Unlabeled point indices | e.g., {501,...,1000} |
| $y_i \in \{0,1\}$ | Ground truth label | 0 or 1 |
| $\hat{p}_i \in [0,1]$ | Final predicted probability | [0, 1] |
| $g_\theta: \mathbb{R}^m \to [0,1]$ | Aggregation function | e.g., mean |
| $\rho \in [0,1]$ | Trade-off parameter | 0.3 - 0.7 |
| $\lambda > 0$ | Regularization strength | 0.01 - 0.1 |
| $d \in \mathbb{N}$ | Latent dimension | 10 - 50 |

---

## Aggregation Function

Maps reconstructed probabilities to final prediction:

$$\hat{p}_i = g_\theta(\hat{r}_{\cdot i})$$

where $\hat{r}_{\cdot i} = [\hat{r}_{1i}, \ldots, \hat{r}_{mi}]^\top$

### Common Choices

**Simple mean**:
$$g(\hat{r}_{\cdot i}) = \frac{1}{m}\sum_{u=1}^m \hat{r}_{ui}$$

**Weighted mean**:
$$g_w(\hat{r}_{\cdot i}) = \sigma(w^\top \hat{r}_{\cdot i} + b)$$
where $\sigma(z) = 1/(1+e^{-z})$

---

## Confidence Weights

### Purpose
Weight which probability predictions to trust more during factorization.

### Strategies

**1. Certainty-based** (default):
$$c_{ui} = |r_{ui} - 0.5|$$

**2. Label-aware** (for labeled data):
$$c_{ui} = \begin{cases}
r_{ui} & \text{if } y_i = 1 \\
1 - r_{ui} & \text{if } y_i = 0
\end{cases}$$

**3. Calibration-based**:
$$c_{ui} = 1 - \text{Brier}_u = 1 - \frac{1}{N}\sum_j (r_{uj} - y_j)^2$$

**4. Agreement-based**:
$$c_{ui} = 1 - \text{Var}_u(r_{\cdot i})$$

---

## Optimization: Alternating Least Squares

### Update Classifier Factors (fix $Y$)

For each classifier $u$:
$$x_u = \left( Y C_u Y^\top + \lambda I \right)^{-1} Y C_u r_u$$

where:
- $C_u = \text{diag}(c_{u1}, \ldots, c_{un})$
- $r_u = [r_{u1}, \ldots, r_{un}]^\top$

### Update Instance Factors (fix $X$)

For each point $i$:
$$y_i = \left( X C_i X^\top + \lambda I \right)^{-1} X C_i r_i$$

where:
- $C_i = \text{diag}(c_{1i}, \ldots, c_{mi})$
- $r_i = [r_{1i}, \ldots, r_{mi}]^\top$

### Update Aggregator (fix $X, Y$)

Gradient descent on supervised loss:
$$\theta \leftarrow \theta - \eta \nabla_\theta L_{\text{sup}}(X, Y, \theta)$$

---

## Algorithm Summary

```
Input: R (m × n), labels (n,), hyperparameters
Output: X (d × m), Y (d × n), θ

1. Initialize:
   - X ← random(d, m) × 0.01
   - Y ← random(d, n) × 0.01
   - θ ← initialize_aggregator()

2. For epoch = 1 to max_iter:
   a. X ← ALS_update_classifiers(Y, R, C, λ)
   b. Y ← ALS_update_instances(X, R, C, λ)
   c. θ ← gradient_step(θ, X, Y, labels, η)
   d. loss ← compute_combined_loss(ρ)
   e. If converged: break

3. Return X, Y, θ
```

---

## Prediction

### On Training/Test Data (transductive)
Given fitted $X, Y, \theta$:
$$\hat{p}_i = g_\theta(X^\top y_i)$$

### On New Data (inductive)
Given new point $i_{\text{new}}$ with predictions $r_{\cdot, i_{\text{new}}}$:

1. Solve for latent factor:
   $$y_{i_{\text{new}}} = (X C X^\top + \lambda I)^{-1} X C r_{i_{\text{new}}}$$

2. Predict:
   $$\hat{p}_{i_{\text{new}}} = g_\theta(X^\top y_{i_{\text{new}}})$$

---

## Hyperparameter Tuning

### Recommended Ranges

| Parameter | Range | Default | Tuning Strategy |
|-----------|-------|---------|-----------------|
| $\rho$ | [0.3, 0.7] | 0.5 | Grid search |
| $d$ | [10, 50] | 20 | Rule: $d \approx \sqrt{m}$ |
| $\lambda$ | [0.01, 0.1] | 0.01 | Grid search |
| Aggregator | {mean, weighted} | weighted | Compare both |

### Validation Strategy

```python
for rho in [0.3, 0.5, 0.7]:
    for d in [10, 20, 30]:
        for lambda in [0.01, 0.05, 0.1]:
            # Train on L_train ∪ U (transductive)
            # Validate on L_val
            # Select best by validation AUC
```

---

## Connection to Knowledge Distillation

| Aspect | Knowledge Distillation | CF-Ensemble |
|--------|------------------------|-------------|
| **Soft targets** | Teacher predictions $q_t$ | Probability matrix $R$ |
| **Student** | Small neural network | Latent factors $X, Y$ |
| **Soft loss** | $\text{KL}(q_t \| q_s)$ | $\sum c_{ui}(r_{ui} - x_u^\top y_i)^2$ |
| **Hard loss** | $\text{CE}(y_g, q_s)$ | $\sum_{i \in \mathcal{L}} \text{CE}(y_i, g(\hat{r}_{\cdot i}))$ |
| **Combined** | $\rho \cdot T^2 \cdot \text{KL} + (1-\rho) \cdot \text{CE}$ | $\rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$ |
| **Key insight** | Match teacher + match labels | Reconstruct matrix + match labels |

---

## Diagnostic Metrics

### Reconstruction Quality

$$\text{RMSE} = \sqrt{\frac{1}{mn}\sum_{u,i} (r_{ui} - \hat{r}_{ui})^2}$$

Monitor separately on $\mathcal{L}$ and $\mathcal{U}$. Large gap suggests overfitting.

### Prediction Performance

- **ROC-AUC**: Primary metric
- **Brier score**: Calibration quality
- **F1 score**: Balanced accuracy

### Comparison Baseline

Always compare against:
1. Simple averaging: $\frac{1}{m}\sum_u r_{ui}$
2. Best single model: $\max_u \text{AUC}_u$
3. Stacking: Logistic regression on $R$

---

## When to Use CF-Ensemble

### ✅ Good fit when:
- Base models have **complementary errors**
- Moderate labeled data (100-10,000 samples)
- Test set is **known at training time** (transductive)
- Need **interpretable** latent structure

### ❌ Poor fit when:
- Base models are **nearly perfect**
- Very large labeled dataset (stacking better)
- Test distribution **very different** from train
- Need strict **inductive** guarantees

---

## Common Pitfalls

### ❌ Setting $\rho = 1$
Pure reconstruction reproduces errors. Always use $\rho \in [0.3, 0.7]$.

### ❌ Ignoring confidence weights
All predictions treated equally. Use label-aware confidence for $\mathcal{L}$.

### ❌ Too large $d$
Overfitting. Start with $d \approx \sqrt{m}$, tune down if needed.

### ❌ Forgetting regularization
Latent factors explode. Always use $\lambda > 0$.

### ❌ Not checking convergence
Stopped too early or too late. Monitor loss curves.

---

## Checklist for Success

- [ ] Diverse base models (different algorithms/hyperparameters)
- [ ] Reasonable $\rho$ (not 0 or 1)
- [ ] Label-aware confidence for labeled data
- [ ] Validation set for hyperparameter tuning
- [ ] Convergence monitoring (loss curves)
- [ ] Baseline comparisons (averaging, stacking)
- [ ] Latent space visualization (sanity check)
- [ ] Test on ≥3 different datasets

---

## One-Liner Summary

**CF-Ensemble = Collaborative Filtering + Supervised Learning**

We factorize the ensemble's probability matrix to discover latent structure, while using ground truth labels to distinguish signal from noise.

---

## Further Reading

- [Knowledge Distillation Tutorial](methods/knowledge_distillation_tutorial.md)
- [CF-Ensemble Optimization Tutorial](methods/cf_ensemble_optimization_objective_tutorial.md)
- [Class-Weighted Gradients](methods/optimization/class_weighted_gradients.md)
- [Failure Modes](failure_modes/README.md)

---

**Print this page for quick reference during implementation!**
