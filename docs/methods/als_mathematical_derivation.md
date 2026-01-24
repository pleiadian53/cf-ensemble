# Alternating Least Squares (ALS): Complete Mathematical Derivation

**A step-by-step derivation of the closed-form update equations for CF-Ensemble**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Setup](#problem-setup)
3. [Deriving the Classifier Update](#deriving-the-classifier-update)
4. [Deriving the Instance Update](#deriving-the-instance-update)
5. [Complete Algorithm](#complete-algorithm)
6. [Convergence Properties](#convergence-properties)
7. [Implementation Notes](#implementation-notes)

---

## Introduction

Alternating Least Squares (ALS) is an optimization algorithm for matrix factorization that alternates between updating two sets of parameters while holding the other fixed. For CF-Ensemble, we factorize the probability matrix $R$ as:

$$R \approx X^T Y$$

where:
- $X \in \mathbb{R}^{d \times m}$: Classifier latent factors (d-dimensional representation of m classifiers)
- $Y \in \mathbb{R}^{d \times n}$: Instance latent factors (d-dimensional representation of n instances)

The key insight is that when one set of factors is fixed, the optimization becomes **convex** with respect to the other, allowing for closed-form solutions.

---

## Problem Setup

### Objective Function

We want to minimize the **confidence-weighted reconstruction loss** with L2 regularization:

$$\mathcal{L}(X, Y) = \sum_{u=1}^{m} \sum_{i=1}^{n} c_{ui}(r_{ui} - x_u^T y_i)^2 + \lambda(||X||_F^2 + ||Y||_F^2)$$

where:
- $r_{ui}$: Probability from classifier $u$ for instance $i$ (entry of matrix $R$)
- $c_{ui}$: Confidence weight for prediction $r_{ui}$ (entry of matrix $C$)
- $x_u$: Latent vector for classifier $u$ (column of $X$)
- $y_i$: Latent vector for instance $i$ (column of $Y$)
- $\lambda$: Regularization parameter
- $||\cdot||_F$: Frobenius norm

### Notation Summary

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $m$ | scalar | Number of classifiers |
| $n$ | scalar | Number of instances |
| $d$ | scalar | Latent dimension |
| $R$ | $m \times n$ | Probability matrix |
| $C$ | $m \times n$ | Confidence weights |
| $X$ | $d \times m$ | Classifier factors |
| $Y$ | $d \times n$ | Instance factors |
| $x_u$ | $d \times 1$ | Latent vector for classifier $u$ |
| $y_i$ | $d \times 1$ | Latent vector for instance $i$ |

---

## Deriving the Classifier Update

We want to update $X$ (all classifier factors) while holding $Y$ fixed.

### Step 1: Per-Classifier Decomposition

The loss can be decomposed into separate terms for each classifier:

$$\mathcal{L}(X, Y) = \sum_{u=1}^{m} \mathcal{L}_u(x_u) + \lambda||Y||_F^2$$

where the loss for classifier $u$ is:

$$\mathcal{L}_u(x_u) = \sum_{i=1}^{n} c_{ui}(r_{ui} - x_u^T y_i)^2 + \lambda||x_u||^2$$

**Key insight**: Each $x_u$ can be optimized **independently**!

### Step 2: Expand the Squared Term

For a single classifier $u$:

$$\mathcal{L}_u(x_u) = \sum_{i=1}^{n} c_{ui}(r_{ui}^2 - 2r_{ui}x_u^T y_i + x_u^T y_i y_i^T x_u) + \lambda x_u^T x_u$$

Dropping constants (terms without $x_u$):

$$\mathcal{L}_u(x_u) = -2\sum_{i=1}^{n} c_{ui} r_{ui} x_u^T y_i + \sum_{i=1}^{n} c_{ui} x_u^T y_i y_i^T x_u + \lambda x_u^T x_u$$

### Step 3: Rewrite in Matrix Form

Using matrix notation:

$$\mathcal{L}_u(x_u) = -2 x_u^T \left(\sum_{i=1}^{n} c_{ui} r_{ui} y_i\right) + x_u^T \left(\sum_{i=1}^{n} c_{ui} y_i y_i^T\right) x_u + \lambda x_u^T x_u$$

Define:
- $b_u = \sum_{i=1}^{n} c_{ui} r_{ui} y_i = Y C_u r_u$ (where $C_u = \text{diag}(c_{u1}, \ldots, c_{un})$)
- $A_u = \sum_{i=1}^{n} c_{ui} y_i y_i^T = Y C_u Y^T$

Then:

$$\mathcal{L}_u(x_u) = -2 x_u^T b_u + x_u^T A_u x_u + \lambda x_u^T x_u$$

$$= -2 x_u^T b_u + x_u^T (A_u + \lambda I) x_u$$

### Step 4: Take the Gradient

Compute $\nabla_{x_u} \mathcal{L}_u(x_u)$:

$$\frac{\partial \mathcal{L}_u}{\partial x_u} = -2b_u + 2(A_u + \lambda I)x_u$$

**Matrix calculus rules used:**
- $\nabla_x (x^T b) = b$
- $\nabla_x (x^T A x) = (A + A^T)x = 2Ax$ (when $A$ is symmetric)

### Step 5: Set Gradient to Zero

For the optimal $x_u^*$:

$$-2b_u + 2(A_u + \lambda I)x_u^* = 0$$

$$(A_u + \lambda I)x_u^* = b_u$$

$$(Y C_u Y^T + \lambda I)x_u^* = Y C_u r_u$$

### Step 6: Solve for $x_u^*$

$$\boxed{x_u^* = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u}$$

This is the **closed-form ALS update for classifier factors**!

### Matrix Notation

For all classifiers simultaneously:
- $C_u = \text{diag}(C[u, :])$ is the diagonal matrix of confidences for classifier $u$
- $r_u = R[u, :]$ is the vector of probabilities from classifier $u$

**Implementation**:
```python
for u in range(m):
    C_u = np.diag(C[u, :])
    A = Y @ C_u @ Y.T + lambda_reg * np.eye(d)
    b = Y @ C_u @ r_u
    x_u = np.linalg.solve(A, b)  # Solve A @ x_u = b
```

---

## Deriving the Instance Update

By symmetry, we can derive the instance update in the same way.

### Step 1: Per-Instance Decomposition

Fix $X$, optimize $Y$. The loss decomposes as:

$$\mathcal{L}(X, Y) = \sum_{i=1}^{n} \mathcal{L}_i(y_i) + \lambda||X||_F^2$$

where:

$$\mathcal{L}_i(y_i) = \sum_{u=1}^{m} c_{ui}(r_{ui} - x_u^T y_i)^2 + \lambda||y_i||^2$$

### Step 2: Expand and Rewrite

Following the same steps as before:

$$\mathcal{L}_i(y_i) = -2 y_i^T \left(\sum_{u=1}^{m} c_{ui} r_{ui} x_u\right) + y_i^T \left(\sum_{u=1}^{m} c_{ui} x_u x_u^T\right) y_i + \lambda y_i^T y_i$$

Define:
- $b_i = \sum_{u=1}^{m} c_{ui} r_{ui} x_u = X C_i r_i$ (where $C_i = \text{diag}(c_{1i}, \ldots, c_{mi})$)
- $A_i = \sum_{u=1}^{m} c_{ui} x_u x_u^T = X C_i X^T$

Then:

$$\mathcal{L}_i(y_i) = -2 y_i^T b_i + y_i^T (A_i + \lambda I) y_i$$

### Step 3: Take Gradient and Solve

$$\frac{\partial \mathcal{L}_i}{\partial y_i} = -2b_i + 2(A_i + \lambda I)y_i = 0$$

$$(A_i + \lambda I)y_i^* = b_i$$

$$(X C_i X^T + \lambda I)y_i^* = X C_i r_i$$

### Step 4: Closed-Form Solution

$$\boxed{y_i^* = (X C_i X^T + \lambda I)^{-1} X C_i r_i}$$

This is the **closed-form ALS update for instance factors**!

### Matrix Notation

For all instances simultaneously:
- $C_i = \text{diag}(C[:, i])$ is the diagonal matrix of confidences for instance $i$
- $r_i = R[:, i]$ is the vector of probabilities for instance $i$

**Implementation**:
```python
for i in range(n):
    C_i = np.diag(C[:, i])
    A = X @ C_i @ X.T + lambda_reg * np.eye(d)
    b = X @ C_i @ r_i
    y_i = np.linalg.solve(A, b)  # Solve A @ y_i = b
```

---

## Complete Algorithm

### Pseudocode

```
Initialize X, Y randomly

for iteration in 1 to max_iter:
    # Update classifier factors
    for u in 1 to m:
        x_u ← (Y C_u Y^T + λI)^(-1) Y C_u r_u
    
    # Update instance factors
    for i in 1 to n:
        y_i ← (X C_i X^T + λI)^(-1) X C_i r_i
    
    # Check convergence
    loss ← compute_loss(X, Y, R, C, λ)
    if |loss_prev - loss| < tol:
        break
```

### Convergence Check

Compute the full reconstruction loss:

$$\mathcal{L}(X, Y) = \sum_{u,i} c_{ui}(r_{ui} - x_u^T y_i)^2 + \lambda(||X||_F^2 + ||Y||_F^2)$$

---

## Convergence Properties

### Guaranteed Decrease

**Theorem**: Each ALS update decreases (or maintains) the objective function value.

**Proof sketch**:
1. When we fix $Y$ and optimize $X$, we solve $\nabla_X \mathcal{L} = 0$, finding the global minimum w.r.t. $X$ (convex problem)
2. Therefore: $\mathcal{L}(X_{\text{new}}, Y) \leq \mathcal{L}(X_{\text{old}}, Y)$
3. Similarly for $Y$ update: $\mathcal{L}(X, Y_{\text{new}}) \leq \mathcal{L}(X, Y_{\text{old}})$

**Implication**: ALS is **guaranteed to converge** to a local minimum (or saddle point).

### Convergence Rate

ALS typically exhibits:
- **Fast initial convergence**: Large loss decrease in first few iterations
- **Slow final convergence**: Linear rate near optimum

**Rule of thumb**: 
- 10-20 iterations usually sufficient
- Monitor relative change: $\frac{|\mathcal{L}_t - \mathcal{L}_{t-1}|}{|\mathcal{L}_t|} < 10^{-4}$

### When ALS Can Fail

ALS is guaranteed to converge, but may get stuck at:
- **Local minima**: Non-convex problem, multiple solutions
- **Saddle points**: Rare but possible
- **Bad initialization**: Random init may start far from good solution

**Solutions**:
- Multiple random restarts
- Smart initialization (e.g., SVD)
- Sufficient regularization ($\lambda > 0$)

---

## Implementation Notes

### Numerical Stability

**Issue**: The matrix $(Y C_u Y^T + \lambda I)$ might be ill-conditioned.

**Solution**: Use `np.linalg.solve` instead of explicit inverse:
```python
# Bad: x_u = inv(A) @ b  (numerically unstable)
# Good: x_u = solve(A, b)  (uses LU decomposition)
x_u = np.linalg.solve(A, b)
```

### Computational Complexity

**Per iteration**:
- Classifier update: $O(md^3)$ (m linear systems of size $d \times d$)
- Instance update: $O(nd^3)$ (n linear systems of size $d \times d$)
- **Total**: $O((m+n)d^3)$ per iteration

**Bottleneck**: Usually instance update since $n \gg m$.

### Memory Efficiency

**Key observation**: We don't need to form $C_u$ or $C_i$ explicitly!

**Efficient computation**:
```python
# Instead of: A = Y @ diag(c_u) @ Y.T
# Use broadcasting: A = (Y * c_u) @ Y.T
Y_weighted = Y * c_u[None, :]  # (d, n)
A = Y_weighted @ Y.T  # (d, d)
```

This avoids creating $n \times n$ or $m \times m$ diagonal matrices.

### Vectorization Opportunities

Current implementation loops over classifiers/instances. For **large-scale problems**, can vectorize:

**Batched solves**:
```python
# Solve multiple systems simultaneously
# Requires: batched linear solver or iterative methods
X = solve_batch([A_1, ..., A_m], [b_1, ..., b_m])
```

**Potential speedup**: 5-10x for $m, n > 1000$

---

## Comparison with Standard Matrix Factorization

### Standard CF (Uniform Weights)

When $c_{ui} = 1$ for all $(u,i)$:

$$x_u^* = (YY^T + \lambda I)^{-1} Y r_u$$

**Simplification**: $YY^T$ is constant across all $u$, compute once!

### Weighted CF (Our Case)

When $c_{ui}$ varies:

$$x_u^* = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u$$

**Cost**: Must recompute $Y C_u Y^T$ for each classifier $u$.

**Why worth it?**  
Confidence weighting allows us to:
- Trust certain predictions more
- Handle uncertainty
- Integrate learned reliability (Phase 3)

---

## Summary of Key Equations

### Classifier Update
$$\boxed{x_u = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u}$$

where:
- $C_u = \text{diag}(c_{u1}, \ldots, c_{un})$
- $r_u = [r_{u1}, \ldots, r_{un}]^T$

### Instance Update
$$\boxed{y_i = (X C_i X^T + \lambda I)^{-1} X C_i r_i}$$

where:
- $C_i = \text{diag}(c_{1i}, \ldots, c_{mi})$
- $r_i = [r_{1i}, \ldots, r_{mi}]^T$

### Matrix Forms

**Gram matrices**:
- $A_u = Y C_u Y^T + \lambda I \in \mathbb{R}^{d \times d}$
- $A_i = X C_i X^T + \lambda I \in \mathbb{R}^{d \times d}$

**Target vectors**:
- $b_u = Y C_u r_u \in \mathbb{R}^d$
- $b_i = X C_i r_i \in \mathbb{R}^d$

**Linear systems**:
- $A_u x_u = b_u$
- $A_i y_i = b_i$

---

## Further Reading

### Foundational Papers
1. **Hu, Koren, Volinsky (2008)**: "Collaborative Filtering for Implicit Feedback Datasets"
   - Introduced confidence-weighted matrix factorization
   - Derived ALS updates for implicit feedback

2. **Zhou et al. (2008)**: "Large-scale Parallel Collaborative Filtering"
   - Scalable ALS implementation
   - Distributed computing considerations

### Extensions
- **Non-negative MF**: Add constraints $X, Y \geq 0$
- **Bayesian MF**: Probabilistic interpretation
- **Neural extensions**: Replace linear $x_u^T y_i$ with neural networks

### Implementation References
- **Spark MLlib**: Production ALS implementation
- **Implicit library**: Python library for implicit feedback CF
- **Surprise**: Scikit-learn-like API for recommender systems

---

## Exercises

### Exercise 1: Verify the Gradient
Manually compute $\nabla_{x_u} \mathcal{L}_u(x_u)$ and verify it matches our derivation.

### Exercise 2: Implement from Scratch
Implement the ALS updates without looking at the code. Compare results.

### Exercise 3: Convergence Analysis
Plot loss vs iteration for different $\lambda$ values. What do you observe?

### Exercise 4: Vectorize
Implement a batched version that solves multiple systems simultaneously.

---

## Conclusion

The ALS update equations:

$$x_u = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u$$
$$y_i = (X C_i X^T + \lambda I)^{-1} X C_i r_i$$

are derived by:
1. Fixing one set of factors
2. Expanding the loss for each factor
3. Taking the gradient
4. Setting to zero and solving

This gives us **closed-form** solutions with **guaranteed convergence** - the key advantage of ALS!

**Next**: See `als_vs_pytorch.md` for a comparison with gradient descent approaches.
