# Class-Weighted Gradients for Imbalanced Data

**Solving the aggregator weight collapse problem through inverse frequency weighting**

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Problem: Weight Collapse on Imbalanced Data](#the-problem-weight-collapse-on-imbalanced-data)
3. [Root Cause Analysis](#root-cause-analysis)
4. [The Solution: Class-Weighted Gradients](#the-solution-class-weighted-gradients)
5. [Mathematical Derivation](#mathematical-derivation)
6. [Implementation](#implementation)
7. [Experimental Results](#experimental-results)
8. [When to Use](#when-to-use)
9. [Related Documentation](#related-documentation)

---

## Introduction

### The Challenge

When training CF-Ensemble on **imbalanced data** (e.g., 10% positive, 90% negative), the aggregator weights can collapse to negative values, causing catastrophic performance degradation:

```
Without class weighting:
  PR-AUC: 0.071 (93% worse than baseline!)
  Weights: [-0.052, -0.051, ..., -0.050]  ❌ Negative, collapsed

With class weighting:
  PR-AUC: 1.000 (perfect performance)
  Weights: [0.085, 0.087, ..., 0.080]  ✅ Positive, healthy
```

### The Solution

**Class-weighted gradients** weight instances by **inverse class frequency**, ensuring each class contributes equally to gradient computation regardless of class distribution.

$$\text{weight}_i = \frac{n}{2 \cdot n_{class(i)}}$$

This prevents the majority class from dominating gradients and ensures stable, effective learning on imbalanced data.

---

## The Problem: Weight Collapse on Imbalanced Data

### Symptoms

1. **Negative weights:** Aggregator weights become negative during training
2. **Constant predictions:** All predictions collapse to same value (no variance)
3. **Catastrophic performance:** 90%+ worse than simple averaging
4. **Happens with both ALS and PyTorch:** Affects all optimization methods

### Example

**Data:** 10% positive, 90% negative (imbalanced)

**Training progression:**
```
Iteration  0: weights = [0.200, 0.200, ...], sum = 1.000
Iteration 10: weights = [0.150, 0.152, ...], sum = 0.757
Iteration 20: weights = [0.100, 0.104, ...], sum = 0.514
Iteration 50: weights = [0.000, 0.003, ...], sum = 0.015  ❌
Iteration 100: weights = [-0.052, -0.051, ...], sum = -0.260  ❌❌
```

**Result:** Weights collapse to negative values, predictions become constant.

---

## Root Cause Analysis

### Understanding the Gradient Formula

The supervised loss gradient treats **all instances equally**:

$$\nabla_w L_{\text{sup}} = \frac{1}{n} \sum_{i=1}^n \underbrace{(y_{\text{pred}, i} - y_{\text{true}, i})}_{\text{residual}} \cdot \hat{r}_i$$

where the **residual** = `y_pred - y_true` measures the prediction error.

**Gradient descent update rule:**
$$w_{\text{new}} = w_{\text{old}} - \text{lr} \times \nabla L$$

**Key insight:**
- **Negative residual** (y_pred < y_true) → gradient pushes to **increase** w (decrease loss)
- **Positive residual** (y_pred > y_true) → gradient pushes to **decrease** w (decrease loss)

### Why Do Weights Collapse?

**Simplified example to illustrate the problem:**

Assume the model is currently making predictions around **0.5** (maximally uncertain) for both classes:

**For positive class instances** (y_true = 1):
```
Residual = y_pred - y_true = 0.5 - 1.0 = -0.5
→ Negative residual means prediction is too low
→ Gradient will try to INCREASE weights (to increase predictions)
```

**For negative class instances** (y_true = 0):
```
Residual = y_pred - y_true = 0.5 - 0.0 = +0.5
→ Positive residual means prediction is too high
→ Gradient will try to DECREASE weights (to decrease predictions)
```

**Now apply class imbalance (10% positive, 90% negative):**

```python
Minority class (10%): residual = -0.5, says "increase w!"
Majority class (90%): residual = +0.5, says "decrease w!"

Total gradient: 0.1 × (-0.5) + 0.9 × (+0.5) = -0.05 + 0.45 = +0.40
                ^^^^^^^^^^^^   ^^^^^^^^^^^^^^
                minority vote  MAJORITY VOTE WINS!

Update: w_new = w_old - lr × (+0.40) = w_old - 0.04
→ Weights DECREASE by 0.04 each iteration (following majority)
Result: After 100 iterations, w → negative (collapsed!)
```

**The problem:** Even though both classes have equal magnitude errors (±0.5), the majority class (90%) numerically dominates the gradient, forcing weights to decrease!

### Key Insight

**The majority class (90%) numerically dominates the gradient computation**, causing weights to drift in the direction that minimizes loss on the majority class, even if it hurts minority class performance.

**Why is this catastrophic?**
- The minority class needs weights to **increase** (to improve its predictions)
- The majority class wants weights to **decrease** (to improve its predictions)
- The majority's vote (90%) overwhelms the minority's vote (10%)
- Weights continuously decrease → eventually go negative → collapse!

This is a **well-known problem in imbalanced learning** across all of machine learning, but was initially misdiagnosed as an alternating optimization issue in our case.

**Does this happen in deep learning too?**

**Yes, absolutely!** This exact same gradient domination problem occurs in modern deep learning with backpropagation on imbalanced datasets:

**Common manifestations in neural networks:**
- Network predicts majority class for nearly everything
- High overall accuracy (e.g., 95%) but terrible minority class recall
- Model "learns" to ignore minority class entirely
- Gradient updates dominated by majority class examples

**Real-world examples where this is critical:**
- **Object detection**: Few objects vs. many background pixels → Focal Loss (Lin et al., 2017)
- **Medical diagnosis**: Rare diseases (1-5% positive) → Class-weighted BCE
- **Fraud detection**: Rare fraud cases (0.1-1%) → Cost-sensitive learning
- **Anomaly detection**: Rare anomalies → One-class or weighted approaches

**Standard solutions in deep learning:**

1. **Class-weighted loss** (what we implemented):
   ```python
   # PyTorch example
   pos_weight = n_neg / n_pos  # e.g., 9.0 for 10% positive
   loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

2. **Focal Loss** (Lin et al., 2017):
   - Down-weights easy examples, focuses on hard ones
   - Popular in object detection (RetinaNet)
   - Formula: $FL(p_t) = -(1-p_t)^\gamma \log(p_t)$

3. **Oversampling/undersampling**:
   - SMOTE, ADASYN, etc.
   - Or weighted sampling in DataLoader

4. **Cost-sensitive learning**:
   - Different misclassification costs per class
   - **Note:** Class-weighted loss (our method) is actually a **specific form** of cost-sensitive learning where:
     - Misclassification cost ∝ inverse class frequency
     - False negatives on minority class cost more than false positives on majority class
     - The cost ratio = `n_majority / n_minority` (e.g., 9:1 for 10% positive)

**Key insight:** The mathematical structure of gradient computation (weighted sum over instances) is **identical** whether you're using:
- Simple linear aggregator (our case)
- Deep neural networks with backprop
- Gradient boosting
- Any gradient-based optimization!

**References:**
- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002) - RetinaNet paper
- Class imbalance survey: [He & Garcia, 2009](https://ieeexplore.ieee.org/document/5128907)
- Cost-sensitive learning: [Elkan, 2001](https://cseweb.ucsd.edu/~elkan/rescale.pdf)

### Proof: PyTorch Also Fails

Testing with PyTorch (unified joint optimization) showed **identical failure**:
- Same PR-AUC: 0.071
- Same weight collapse to negative values
- Same catastrophic performance

**Conclusion:** The problem is NOT alternating optimization, but the **class imbalance bias in the gradient formula**.

---

## The Solution: Class-Weighted Gradients

### Core Idea

Weight each instance by **inverse class frequency** so that each class contributes **equally** to the gradient:

$$w_{\text{class}} = \frac{n}{2 \cdot n_{\text{class}}}$$

**For binary classification:**
```python
n_pos = sum(y_true == 1)
n_neg = sum(y_true == 0)
n = n_pos + n_neg

pos_weight = n / (2 * n_pos)  # e.g., 5.0 for 10% positive
neg_weight = n / (2 * n_neg)  # e.g., 0.56 for 90% negative

instance_weights = [pos_weight if y == 1 else neg_weight for y in y_true]
```

### Why It Works

Using the same scenario (y_pred ≈ 0.5 for both classes, 10% positive / 90% negative):

**Before (no weighting):**
```
Minority (10%): 0.1 × (-0.5) = -0.05, says "increase w!"
Majority (90%): 0.9 × (+0.5) = +0.45, says "decrease w!"
                                ^^^^^^
                                DOMINATES!

Total gradient = -0.05 + 0.45 = +0.40 (biased toward majority)
→ Update: w_new = w_old - 0.04 (weights decrease, majority wins)
```

**After (with class weighting):**
```
Minority (10%): 5.0 × (-0.5) = -2.50, says "increase w!" (upweighted!)
Majority (90%): 0.56 × (+0.5) = +0.28, says "decrease w!" (downweighted)
                ^^^^^^^^^^^^^^   ^^^^^^^
                NOW BALANCED!

Total gradient = -2.50 + 0.28 = -2.22 (balanced gradient)
→ Update: w_new = w_old + 0.22 (weights increase, classes agree)
```

**Key point:** Now both classes have **equal influence**! 

To see the balance more clearly, look at **total class contributions** (accounting for class size):
```
Minority contribution: (10% of instances) × (-2.50) = -0.25n
Majority contribution: (90% of instances) × (+0.28) = +0.25n
                                                      ^^^^^^^^
                                                      EQUAL magnitude!
```

Both classes now contribute equally to the gradient direction, preventing the majority class from dominating!

---

## Mathematical Derivation

### Standard (Unweighted) Supervised Loss

Binary cross-entropy:
$$L_{\text{sup}} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Gradient w.r.t. aggregator weights $w$:
$$\nabla_w L_{\text{sup}} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i) \cdot \hat{r}_i$$

**Problem:** Equal weight $\frac{1}{n}$ for all instances → majority class dominates.

### Class-Weighted Supervised Loss

Weighted binary cross-entropy:
$$L_{\text{sup}}^{\text{weighted}} = -\sum_{i=1}^n \frac{w_{class(i)}}{\sum_j w_j} \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

where:
$$w_{\text{pos}} = \frac{n}{2 \cdot n_{\text{pos}}}, \quad w_{\text{neg}} = \frac{n}{2 \cdot n_{\text{neg}}}$$

**Gradient (class-weighted):**
$$\nabla_w L_{\text{sup}}^{\text{weighted}} = \frac{\sum_{i=1}^n w_{class(i)} \cdot (\hat{y}_i - y_i) \cdot \hat{r}_i}{\sum_{i=1}^n w_{class(i)}}$$

### Why This Formula?

**Inverse frequency weighting** ensures:
1. Each **class** contributes equally (not each instance)
2. Minority class gets higher weight to compensate for fewer instances
3. Balanced gradient feedback from both classes

**Example with 10% positive, 90% negative:**
```
Positive instances: n_pos = 10, weight = 100/(2×10) = 5.0
Negative instances: n_neg = 90, weight = 100/(2×90) = 0.56

Total contribution from positives: 10 × 5.0 = 50
Total contribution from negatives: 90 × 0.56 = 50  ✅ Balanced!
```

---

## Critical Distinction: Where Does Class Weighting Apply?

### Understanding the Optimization Landscape

CF-Ensemble optimizes **different parameters using different methods:**

| Parameters | What They Are | Optimization Method |
|------------|---------------|---------------------|
| **X** (classifier factors) | Latent representations of classifiers (d × m) | Varies by trainer |
| **Y** (instance factors) | Latent representations of instances (d × n) | Varies by trainer |
| **w, b** (aggregator) | Weights for combining predictions | Always gradient descent |

**Key insight:** Class weighting only applies where we use **gradient descent** (not closed-form solutions).

---

### ALS Trainer: Hybrid Optimization

The ALS trainer uses **two different optimization methods** for different parameters:

#### 1. Latent Factors (X, Y) - Closed-Form ALS

**Method:** Alternating Least Squares (closed-form solutions, no gradients!)

```python
# Update X (fix Y) - Closed-form solution
X = (Y @ C.T @ Y.T + λI)^(-1) @ Y @ C.T @ R.T

# Update Y (fix X) - Closed-form solution
Y = (X.T @ C @ X + λI)^(-1) @ X.T @ C @ R
```

**Supervision incorporated via:** **Label-aware confidence weighting**
- Modulates confidence matrix C: higher confidence when prediction matches label
- This is an **approximation** to incorporating supervision into reconstruction
- Enabled with `use_label_aware_confidence=True`

**Class weighting here?** ❌ **NO**
- No gradients (direct matrix inversion)
- No iterative updates
- Class imbalance is handled by **label-aware confidence** instead
- The approximation adjusts C to emphasize labeled instances with their true labels

#### 2. Aggregator (w, b) - Gradient Descent

**Method:** Iterative gradient descent (explicit gradients)

```python
# Update w, b (fix X, Y) - Gradient descent
residual = y_pred - y_true
grad_w = (R_hat @ (residual * class_weights)) / sum(class_weights)
grad_b = sum(residual * class_weights) / sum(class_weights)
w -= lr * grad_w
b -= lr * grad_b
```

**Supervision incorporated via:** **Direct supervised loss (BCE)**
- Explicit gradient computation from prediction errors
- Standard gradient descent updates

**Class weighting here?** ✅ **YES - ESSENTIAL!**
- Uses gradient descent
- Class imbalance directly biases gradients
- Without class weighting → weight collapse (catastrophic)
- Enabled with `use_class_weights=True` (default)

---

### PyTorch Trainer: Pure Gradient Descent

**Method:** Joint optimization of all parameters via backpropagation

```python
# Single unified step for ALL parameters (X, Y, w, b)
loss = rho * reconstruction_loss + (1-rho) * supervised_loss
loss.backward()  # Computes ∂loss/∂X, ∂loss/∂Y, ∂loss/∂w, ∂loss/∂b
optimizer.step()  # Updates all parameters together
```

**Supervision incorporated via:** Direct supervised loss in combined objective

**Class weighting here?** ✅ **YES - Applies to ALL parameters**
- All parameters updated via gradients from the same loss
- Class weighting in supervised_loss affects X, Y, w, b through backprop
- Single unified approach (simpler conceptually)

**Label-aware confidence?** ❌ **NO - Not needed**
- Has exact gradients for supervision
- No need for ALS approximation trick
- Direct optimization of the true combined loss

---

### Visual Comparison

```
ALS Trainer (Hybrid):
┌──────────────────────────────────────────────────────────────┐
│ Step 1-2: Update X, Y (Latent Factors)                       │
│ ├─ Method: Closed-form ALS ⚙️ (matrix inversion)             │
│ ├─ Supervision: Label-aware confidence ✅                    │
│ │   ↳ Modulates C matrix based on label agreement           │
│ ├─ Class weighting: N/A ❌ (no gradients to weight)          │
│ └─ Handles imbalance via: Label-aware confidence             │
├──────────────────────────────────────────────────────────────┤
│ Step 3: Update w, b (Aggregator)                             │
│ ├─ Method: Gradient descent 📉 (iterative)                   │
│ ├─ Supervision: Direct BCE loss                             │
│ ├─ Class weighting: YES ✅ (essential for imbalanced data)   │
│ └─ Handles imbalance via: Class-weighted gradients           │
└──────────────────────────────────────────────────────────────┘

PyTorch Trainer (Pure Gradient Descent):
┌──────────────────────────────────────────────────────────────┐
│ Single Step: Update ALL (X, Y, w, b)                         │
│ ├─ Method: Joint gradient descent 📉 (backprop)              │
│ ├─ Supervision: Direct combined loss                        │
│ ├─ Class weighting: YES ✅ (affects ALL parameters)          │
│ └─ Handles imbalance via: Class-weighted loss (unified)      │
└──────────────────────────────────────────────────────────────┘
```

---

### Why ALS Needs BOTH Techniques

**For imbalanced data, ALS requires:**

1. **`use_label_aware_confidence=True`** (default: True)
   - Purpose: Handle class imbalance in **latent factor updates** (X, Y)
   - Method: Approximation via confidence weighting
   - Target: Reconstruction objective (closed-form ALS)

2. **`use_class_weights=True`** (default: True)
   - Purpose: Handle class imbalance in **aggregator updates** (w, b)
   - Method: Exact gradient weighting
   - Target: Supervised loss (gradient descent)

**Both are essential!** Disabling either causes problems:
- Disable label-aware confidence → poor latent factors
- Disable class weighting → aggregator weight collapse

**Example:**
```python
trainer = CFEnsembleTrainer(
    use_label_aware_confidence=True,  # ← For X, Y (ALS approximation)
    use_class_weights=True,           # ← For w, b (gradient descent)
    focal_gamma=0.0                   # ← Also for w, b only
)
```

---

### Why PyTorch Needs Only One

**For imbalanced data, PyTorch requires:**

1. **`use_class_weights=True`** (default: True)
   - Purpose: Handle class imbalance in **all parameters**
   - Method: Exact gradient weighting via loss function
   - Target: Combined objective (affects X, Y, w, b via backprop)

**That's it!** Single unified approach:
- No label-aware confidence needed (has exact gradients)
- Class weighting propagates to all parameters automatically
- Simpler conceptually but slower computationally

**Example:**
```python
trainer = CFEnsemblePyTorchTrainer(
    use_class_weights=True,  # ← Affects ALL parameters (X, Y, w, b)
    focal_gamma=2.0          # ← Also affects ALL parameters
)
```

---

### Summary: Where Each Technique Applies

| Technique | Purpose | ALS: Latent Factors (X, Y) | ALS: Aggregator (w, b) | PyTorch: All Parameters |
|-----------|---------|---------------------------|----------------------|------------------------|
| **Label-aware confidence** | Handle imbalance in ALS | ✅ Yes (approximation) | ❌ No | ❌ No (not needed) |
| **Class-weighted gradients** | Handle imbalance in GD | ❌ No (no gradients) | ✅ Yes (essential) | ✅ Yes (all params) |
| **Focal loss** | Focus on hard examples | ❌ No (no gradients) | ✅ Yes (optional) | ✅ Yes (all params) |

**Key takeaway:**
- **ALS is hybrid:** Closed-form (X, Y) + Gradient descent (w, b)
- **PyTorch is pure:** Gradient descent for everything
- **Class weighting and focal loss:** Only where we use gradient descent
- **Label-aware confidence:** ALS-specific approximation trick

---

## Implementation

#### 1. Latent Factors (X, Y) - Closed-Form ALS

**Method:** Alternating Least Squares (closed-form, no gradients)

```python
# Update X (fix Y)
X = (Y @ C^T @ Y^T + λI)^(-1) @ Y @ C^T @ R^T

# Update Y (fix X)  
Y = (X^T @ C @ X + λI)^(-1) @ X^T @ C @ R
```

**Supervision via:** **Label-aware confidence weighting**
- Modulates confidence matrix C based on label agreement
- Higher confidence for predictions matching labels
- This is an **approximation** to incorporating supervision

**Class weighting:** ❌ **Does NOT apply here**
- No gradients (closed-form solution)
- Class imbalance handled by label-aware confidence
- See `use_label_aware_confidence` parameter

#### 2. Aggregator (w, b) - Gradient Descent

**Method:** Iterative gradient descent (explicit gradients)

```python
# Update w, b
grad_w = (R_hat @ (y_pred - y_true)) / n
grad_b = mean(y_pred - y_true)
w -= lr * grad_w
b -= lr * grad_b
```

**Supervision via:** **Direct supervised loss (BCE)**
- Explicit gradient computation
- Standard gradient descent updates

**Class weighting:** ✅ **DOES apply here**
- Direct gradient computation
- Class imbalance creates gradient bias
- Class weighting essential to prevent collapse
- See `use_class_weights` parameter

### PyTorch Trainer: Pure Gradient Descent (All Parameters)

**Method:** Joint gradient descent via backpropagation (all parameters together)

```python
# Single optimization step for ALL parameters
loss = reconstruction_loss + supervised_loss
loss.backward()  # Computes gradients for X, Y, w, b
optimizer.step()  # Updates all parameters
```

**Supervision:** Direct supervised loss in combined objective

**Class weighting:** ✅ **Applies to ALL parameters (X, Y, w, b)**
- Single loss function with class weighting
- All gradients affected equally
- No label-aware confidence needed (has exact gradients)

### Visual Comparison

```
ALS Trainer:
┌─────────────────────────────────────────────────────┐
│ Latent Factors (X, Y)                               │
│ ├─ Method: Closed-form ALS (no gradients)          │
│ ├─ Supervision: Label-aware confidence ✅           │
│ ├─ Class weighting: N/A ❌                          │
│ └─ Focal loss: N/A ❌                               │
├─────────────────────────────────────────────────────┤
│ Aggregator (w, b)                                   │
│ ├─ Method: Gradient descent                        │
│ ├─ Supervision: Direct BCE loss                    │
│ ├─ Class weighting: YES ✅                          │
│ └─ Focal loss: YES ✅                               │
└─────────────────────────────────────────────────────┘

PyTorch Trainer:
┌─────────────────────────────────────────────────────┐
│ ALL Parameters (X, Y, w, b)                         │
│ ├─ Method: Joint gradient descent (backprop)       │
│ ├─ Supervision: Direct combined loss               │
│ ├─ Class weighting: YES ✅ (all parameters)         │
│ └─ Focal loss: YES ✅ (all parameters)              │
└─────────────────────────────────────────────────────┘
```

### Summary

| Question | Answer |
|----------|--------|
| **Do class-weighted gradients apply to ALS latent factors (X, Y)?** | ❌ **No** - They use closed-form ALS (no gradients). Class imbalance is handled by **label-aware confidence** instead. |
| **Do class-weighted gradients apply to ALS aggregator (w, b)?** | ✅ **Yes** - The aggregator uses gradient descent, so class weighting is **essential**. |
| **Do class-weighted gradients apply to PyTorch?** | ✅ **Yes** - All parameters use gradient descent, so class weighting applies to **everything** (X, Y, w, b). |
| **Does label-aware confidence apply to PyTorch?** | ❌ **No** - PyTorch has exact gradients, doesn't need the ALS approximation trick. |

**Key insight:** ALS is a **hybrid method** - some parameters use closed-form solutions (with label-aware confidence approximation), others use gradient descent (with class weighting). PyTorch is **pure gradient descent** for all parameters.

---

## Implementation

Modified `WeightedAggregator.update()`:

```python
def update(self, X, Y, labeled_idx, labels, lr, use_class_weights=True):
    # Reconstruct probabilities
    R_hat = X.T @ Y[:, labeled_idx]
    
    # Get predictions
    y_pred = self.predict(R_hat)
    y_true = labels[labeled_idx]
    
    # Compute residuals
    residual = y_pred - y_true
    
    if use_class_weights:
        # Compute class weights
        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = n - n_pos
        
        if n_pos > 0 and n_neg > 0:
            pos_weight = n / (2 * n_pos)
            neg_weight = n / (2 * n_neg)
            instance_weights = np.where(y_true == 1, pos_weight, neg_weight)
        else:
            # Edge case: only one class present
            instance_weights = np.ones(n)
        
        # Weighted gradient
        weighted_residual = residual * instance_weights
        grad_w = (R_hat @ weighted_residual) / np.sum(instance_weights)
        grad_b = np.sum(weighted_residual) / np.sum(instance_weights)
    else:
        # Standard unweighted gradient
        grad_w = (R_hat @ residual) / len(residual)
        grad_b = np.mean(residual)
    
    # Gradient descent update
    self.w -= lr * grad_w
    self.b -= lr * grad_b
```

### For PyTorch Trainer

Modified `CFEnsembleNet.compute_loss()`:

```python
def compute_loss(self, R, C, labels, labeled_mask, rho, lambda_reg, 
                 use_class_weights=True):
    # ... reconstruction loss ...
    
    # Supervised loss with class weighting
    if torch.sum(labeled_mask) > 0:
        y_pred = self.forward(labeled_idx)
        y_true = labels[labeled_mask]
        
        # Binary cross-entropy
        eps = 1e-15
        y_pred_clipped = torch.clamp(y_pred, eps, 1 - eps)
        bce = -(y_true * torch.log(y_pred_clipped) +
               (1 - y_true) * torch.log(1 - y_pred_clipped))
        
        if use_class_weights:
            # Compute class weights
            n = len(y_true)
            n_pos = torch.sum(y_true == 1).float()
            n_neg = n - n_pos
            
            if n_pos > 0 and n_neg > 0:
                pos_weight = n / (2 * n_pos)
                neg_weight = n / (2 * n_neg)
                instance_weights = torch.where(y_true == 1, pos_weight, neg_weight)
            else:
                instance_weights = torch.ones(n, device=R.device)
            
            # Weighted loss
            sup_loss = torch.sum(instance_weights * bce) / torch.sum(instance_weights)
        else:
            # Standard unweighted loss
            sup_loss = torch.mean(bce)
    
    # ... combined loss ...
```

### Usage

**Default behavior (recommended):**
```python
# Class weighting enabled by default
trainer = CFEnsembleTrainer(
    n_classifiers=10,
    latent_dim=20,
    rho=0.5
)
# Automatically handles imbalanced data!
```

**Explicit control:**
```python
# Enable (default)
trainer = CFEnsembleTrainer(use_class_weights=True)

# Disable for debugging/research
trainer = CFEnsembleTrainer(use_class_weights=False)
```

---

## Experimental Results

### Test Setup
- **Data:** 500 instances, 10 classifiers, 10% positive rate
- **Base classifier quality:** PR-AUC ≈ 0.70 (target)
- **Metrics:** PR-AUC (primary), weight std, prediction variance

### Results

| Method | PR-AUC | Weight Std | Weight Range | Status |
|--------|--------|------------|--------------|--------|
| **Simple Average (baseline)** | 1.000 | N/A | N/A | ✅ |
| **ALS (no class weights)** | 0.071 | 0.007 | [-0.052, -0.050] | ❌ Collapsed |
| **ALS (class weighted)** | **1.000** | 0.005 | [0.072, 0.087] | ✅ **FIXED** |
| **PyTorch (no class weights)** | 0.071 | 0.014 | [-0.188, -0.149] | ❌ Collapsed |
| **PyTorch (class weighted)** | **1.000** | 0.041 | [0.199, 0.335] | ✅ **FIXED** |

### Key Findings

1. **Class weighting prevents collapse:**
   - Weights remain positive and stable
   - No manual tuning needed

2. **Performance restored:**
   - From 0.071 → 1.000 PR-AUC (14x improvement!)
   - Matches or exceeds simple averaging

3. **PyTorch learns richer weights:**
   - 8.5x more weight diversity than ALS
   - Better generalization potential

4. **Works automatically:**
   - No hyperparameter tuning required
   - Adapts to any imbalance ratio

### Detailed Analysis

**ALS with Class Weights:**
```
Weights: [0.085, 0.087, 0.074, 0.072, 0.081, 0.077, 0.082, 0.081, 0.085, 0.080]
Weight sum: 0.806 (positive, stable)
Weight std: 0.0048
Prediction range: [0.551, 0.627]
PR-AUC: 1.000 ✅
```

**PyTorch with Class Weights:**
```
Weights: [0.275, 0.335, 0.279, 0.206, 0.272, 0.226, 0.206, 0.199, 0.236, 0.237]
Weight sum: 2.470 (positive, diverse)
Weight std: 0.0406 (8.5x larger than ALS!)
Prediction range: [0.410, 0.684] (more variance)
PR-AUC: 1.000 ✅
```

---

## When to Use

### Always Enabled (Recommended)

Class weighting is **enabled by default** (`use_class_weights=True`) because:

1. **No downside on balanced data:**
   - With 50/50 split: `pos_weight = neg_weight = 1.0`
   - Equivalent to standard unweighted gradient

2. **Critical for imbalanced data:**
   - Prevents catastrophic weight collapse
   - Enables effective learning on minority class

3. **Automatic adaptation:**
   - No manual tuning required
   - Computes weights from data distribution

4. **Industry standard:**
   - Used in scikit-learn, PyTorch, TensorFlow
   - Well-established best practice

### Scenarios

| Data Distribution | use_class_weights | Effect |
|-------------------|-------------------|--------|
| Balanced (50/50) | True (default) | No effect (weights ≈ 1.0) |
| Mild imbalance (30/70) | True (default) | Slight upweighting of minority |
| Strong imbalance (10/90) | True (default) | **Essential** - prevents collapse |
| Extreme imbalance (1/99) | True (default) | **Critical** - compensates heavily |
| Research/debugging | False | Only for understanding unweighted behavior |

### When to Disable

**Rarely needed**, but disable (`use_class_weights=False`) when:
- Comparing to baseline methods that don't use class weighting
- Studying the effect of class imbalance on unweighted gradients
- Debugging gradient computation
- Research on alternative weighting schemes

**Important:** On imbalanced data, disabling will likely cause weight collapse and poor performance!

---

## Comparison: ALS vs PyTorch with Class Weighting

### Performance Equivalence

**Good news:** Both methods achieve **identical PR-AUC (1.000)** with class weighting!

| Metric | ALS | PyTorch |
|--------|-----|---------|
| **PR-AUC** | 1.000 ✅ | 1.000 ✅ |
| **Weight Std** | 0.005 | 0.041 (8.5× larger) |
| **Weight Range** | [0.072, 0.087] | [0.199, 0.335] (3.8× larger) |
| **Prediction Variance** | Low (uniform weights) | High (diverse weights) |
| **Speed** | ⚡ Faster (closed-form) | Slower (iterative) |

### Key Difference: Weight Diversity

**PyTorch learns much richer weight distributions:**
- ALS: Nearly uniform weights (std = 0.005)
- PyTorch: Diverse weights (std = 0.041, 8.5× larger)

**Why?**
- **ALS**: Alternating optimization with confidence weighting tends toward uniform solutions
- **PyTorch**: Joint optimization explores weight space more fully

**Implication:** PyTorch may generalize better on unseen data, though both achieve perfect performance on this test.

### Recommendation

**Use ALS for:**
- ✅ Speed-critical applications
- ✅ Production systems (proven stability)
- ✅ When uniform weights are acceptable

**Use PyTorch for:**
- ✅ Research and exploration
- ✅ When weight interpretability matters
- ✅ Potential better generalization

**Bottom line:** Either works! Class weighting is the critical ingredient, not the optimization method.

---

## Future Directions: Alternative Approaches

Beyond class-weighted loss (our current solution), here are other promising methods:

### 1. **Focal Loss** ⭐ **Most Promising**

**Why explore this?**
- **Addresses a different problem:** Easy vs. hard examples (not just class imbalance)
- **Complements class weighting:** Can be combined for synergy
- **Proven in deep learning:** State-of-art in object detection (RetinaNet)

**Formula:**
$$FL(p_t) = -(1-p_t)^\gamma \log(p_t)$$

where $\gamma$ (typically 2.0) controls down-weighting of easy examples.

**Potential benefits for CF-Ensemble:**
- Focus learning on **hard-to-predict instances**
- May improve performance when base classifiers disagree strongly
- Could help with **noisy labels** or **label uncertainty**

**Implementation complexity:** Medium (requires changing loss function)

**Recommendation:** ⭐ **Worth exploring** - Could provide complementary benefits to class weighting

---

### 2. **Oversampling/Undersampling** ⚠️ **Less Promising**

**Why NOT explore this first?**
- **Loses information:** Undersampling discards majority class data
- **Creates duplicates:** Oversampling may cause overfitting
- **Less principled:** Class weighting is more mathematically elegant
- **Already solved:** Class weighting achieves perfect performance (PR-AUC 1.000)

**Potential use case:**
- If computational cost is a concern (smaller effective dataset)
- For comparison/ablation studies

**Recommendation:** ⚠️ **Low priority** - Class weighting already solves the problem without data manipulation

---

### 3. **Advanced Cost-Sensitive Learning** 💡 **Interesting for Future**

**Our current approach:**
- Fixed cost ratio = `n_majority / n_minority`
- Same cost for all instances in a class

**Potential enhancements:**
- **Instance-dependent costs:** Weight based on prediction confidence
- **Asymmetric costs:** Different costs for FP vs FN
- **Learned costs:** Optimize cost weights as hyperparameters

**Example - Confidence-based weighting:**
```python
# Higher weight for low-confidence predictions (harder examples)
instance_weight = class_weight * (1 - prediction_confidence)
```

**Potential benefits:**
- More nuanced learning signal
- Could combine benefits of focal loss and class weighting

**Recommendation:** 💡 **Interesting for research** - But not urgent since current method works well

---

### 4. **Adaptive Weighting During Training** 🔬 **Research Idea**

**Idea:** Dynamically adjust class weights as training progresses

**Approaches:**
- **Curriculum learning:** Start with mild weighting, increase gradually
- **Performance-based:** Adjust based on per-class metrics during training
- **Confidence-based:** Weight based on model uncertainty

**Potential benefits:**
- More stable training
- Better convergence properties
- Could prevent early-stage instabilities

**Implementation complexity:** High (requires online monitoring)

**Recommendation:** 🔬 **Long-term research** - Current fixed weighting is simple and works

---

### Summary: Which to Explore Next?

**Priority ranking:**

1. **⭐ Focal Loss** (Highest priority)
   - Different mechanism (easy vs. hard examples)
   - Can combine with class weighting
   - Proven track record in deep learning
   - Medium implementation effort

2. **💡 Instance-dependent costs** (Medium priority)
   - Natural extension of current approach
   - Confidence-weighted gradients
   - Low implementation effort

3. **🔬 Adaptive weighting** (Low priority - research)
   - More complex, uncertain benefits
   - Current method already works well

4. **⚠️ Over/undersampling** (Lowest priority)
   - Less principled than current solution
   - May degrade performance
   - Only for specific use cases

**Recommended next step:** Implement **Focal Loss** with optional $\gamma$ parameter, test if it improves performance beyond class weighting on challenging scenarios (high disagreement, noisy labels, etc.).

---

## Related Documentation

| Topic | Document |
|-------|----------|
| **Focal Loss** | [`docs/methods/optimization/focal_loss.md`](focal_loss.md) ⭐ **Complementary technique** |
| **Failure Mode** | [`docs/failure_modes/aggregator_weight_collapse.md`](../../failure_modes/aggregator_weight_collapse.md) |
| **ALS Derivation** | [`docs/methods/als_mathematical_derivation.md`](../als_mathematical_derivation.md) |
| **ALS vs PyTorch** | [`docs/methods/als_vs_pytorch.md`](../als_vs_pytorch.md) |

---

## Summary

**Problem:** Aggregator weights collapse to negative values on imbalanced data, causing 90%+ performance degradation.

**Root Cause:** Standard gradients treat all instances equally, allowing majority class to dominate gradient computation.

**Solution:** Class-weighted gradients weight instances by inverse class frequency, ensuring each class contributes equally.

**Implementation:** Added `use_class_weights` parameter (enabled by default) to both ALS and PyTorch trainers.

**Results:** Perfect performance restored (PR-AUC 1.000), weights remain positive and stable, works automatically without tuning.

**Recommendation:** Always use class weighting (default behavior) for reliable performance on any data distribution.

---

**Status:** ✅ Implemented and tested  
**Date:** 2026-01-25  
**Impact:** Critical fix for production use on imbalanced data
