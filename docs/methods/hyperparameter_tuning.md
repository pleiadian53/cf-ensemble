# Hyperparameter Tuning for CF-Ensemble

**How to find optimal hyperparameters for your dataset**

---

## The ρ (Rho) Parameter: Balancing Reconstruction and Supervision

### What is ρ?

The parameter ρ ∈ [0, 1] controls the balance between two competing objectives:

$$\mathcal{L} = \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta)$$

- **ρ = 1.0**: Pure reconstruction (collaborative filtering only)
- **ρ = 0.5**: Balanced (recommended default)
- **ρ = 0.0**: Pure supervised (ignore reconstruction)

---

## Which ρ Should You Use?

### Rule of Thumb

| Scenario | Recommended ρ | Rationale |
|----------|---------------|-----------|
| **Many labels** (>50% labeled) | 0.3 - 0.5 | Supervised signal is strong, focus on it |
| **Balanced** (~50% labeled) | 0.5 | Equal weight to both objectives |
| **Few labels** (<20% labeled) | 0.5 - 0.7 | Leverage reconstruction to learn from unlabeled |
| **Very few labels** (<5%) | 0.7 - 0.9 | Mostly rely on structure, light supervision |
| **No labels** (pure test set) | 1.0 | Pure collaborative filtering |

### Intuition

**Why not always use ρ=0 (pure supervised)?**
- Reconstruction provides **regularization** through the manifold hypothesis
- It leverages **unlabeled data** (transductive learning)
- It helps with **diverse errors**: smooths out individual classifier mistakes

**Why not always use ρ=1 (pure reconstruction)?**
- Reconstruction can **reproduce errors** if all classifiers make the same mistake
- Supervised signal **guides towards correct answers** rather than just consistency
- Without supervision, the system doesn't know which direction to improve

**The sweet spot (ρ ≈ 0.5)**:
- Reconstruction acts as **prior knowledge** ("similar instances should get similar predictions")
- Supervision acts as **correction** ("but these specific instances should be class 1")
- Together they overcome individual limitations

---

## How to Determine ρ in Practice

### Method 1: Cross-Validation (Recommended)

Use validation set performance to select ρ:

```python
from cfensemble.data import EnsembleData
from cfensemble.optimization import CFEnsembleTrainer
from sklearn.metrics import roc_auc_score
import numpy as np

# Create data with validation split
data = EnsembleData(R_train, labels_train)
train_data, val_data = data.split_labeled_data(train_fraction=0.8)

# Grid search over rho
rho_values = [0.0, 0.3, 0.5, 0.7, 1.0]
results = []

for rho in rho_values:
    trainer = CFEnsembleTrainer(
        n_classifiers=R_train.shape[0],
        latent_dim=20,
        rho=rho,
        lambda_reg=0.01,
        max_iter=50,
        verbose=False
    )
    trainer.fit(train_data)
    
    # Evaluate on validation set
    val_pred = trainer.predict(val_data)
    val_labeled_idx = val_data.labeled_idx
    auc = roc_auc_score(
        val_data.labels[val_labeled_idx],
        val_pred[val_labeled_idx]
    )
    
    results.append({'rho': rho, 'val_auc': auc})
    print(f"ρ={rho:.1f}: Val AUC={auc:.4f}")

# Select best rho
best_rho = max(results, key=lambda x: x['val_auc'])['rho']
print(f"\nBest ρ: {best_rho:.1f}")

# Retrain on full labeled data with best rho
final_trainer = CFEnsembleTrainer(
    n_classifiers=R_train.shape[0],
    rho=best_rho,
    latent_dim=20
)
final_trainer.fit(data)
```

**Time complexity**: O(k × training_time) where k is number of ρ values to try.  
**Typical k**: 5-7 values is sufficient (0.0, 0.3, 0.5, 0.7, 0.9)

---

### Method 2: Learning Curve Analysis

Examine how different ρ values affect training:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for rho in [0.0, 0.5, 1.0]:
    trainer = CFEnsembleTrainer(
        n_classifiers=m,
        rho=rho,
        max_iter=50,
        verbose=False
    )
    trainer.fit(data)
    
    # Plot loss curves
    axes[0].plot(trainer.history['loss'], label=f'ρ={rho:.1f}')
    axes[1].plot(trainer.history['reconstruction'], label=f'ρ={rho:.1f}')
    axes[2].plot(trainer.history['supervised'], label=f'ρ={rho:.1f}')

axes[0].set_title('Total Loss')
axes[1].set_title('Reconstruction Loss')
axes[2].set_title('Supervised Loss')
for ax in axes:
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**What to look for**:
- **Smooth convergence**: Good ρ values show steady decrease
- **Balance**: Both loss components should decrease (not one dominating)
- **Overfitting**: If validation loss increases while training decreases, reduce ρ or increase λ

---

### Method 3: Domain Knowledge

Use prior knowledge about your data:

**High ρ (reconstruction-heavy) when**:
- Base classifiers are **diverse** (different algorithms, features, etc.)
- Errors are **uncorrelated** (one wrong doesn't mean all wrong)
- Dataset has **strong structure** (manifold hypothesis holds)
- You have **many unlabeled instances** to learn from

**Low ρ (supervision-heavy) when**:
- Base classifiers are **similar** (same algorithm, different seeds)
- Errors are **correlated** (systematic biases)
- Dataset is **noisy** or lacks structure
- You have **abundant labels**

**Example**: If you have 10 diverse classifiers (random forest, gradient boosting, SVM, neural nets) with 30% labeled data, start with ρ=0.5.

---

## Other Important Hyperparameters

### Latent Dimensionality (d)

**What it controls**: Expressiveness of the latent space.

**Recommended range**: 10-50

| d | When to use |
|---|-------------|
| 5-10 | Small datasets (<1000 instances), simple problems |
| 10-20 | **Default choice**, works for most problems |
| 20-50 | Large datasets (>10,000), complex relationships |
| >50 | Risk of overfitting, rarely needed |

**Selection**:
```python
for d in [10, 20, 30, 40]:
    trainer = CFEnsembleTrainer(n_classifiers=m, latent_dim=d, rho=0.5)
    trainer.fit(train_data)
    # Evaluate on validation set
```

---

### Regularization Strength (λ)

**What it controls**: L2 penalty on latent factors to prevent overfitting.

**Recommended range**: 0.001 - 0.1

| λ | When to use |
|---|-------------|
| 0.001-0.01 | **Default**, large datasets (>5000 instances) |
| 0.01-0.05 | Medium datasets (1000-5000) |
| 0.05-0.1 | Small datasets (<1000), high risk of overfitting |

**Selection** (via validation):
```python
for lambda_reg in [0.001, 0.01, 0.05, 0.1]:
    trainer = CFEnsembleTrainer(
        n_classifiers=m,
        latent_dim=20,
        rho=0.5,
        lambda_reg=lambda_reg
    )
    # Train and evaluate
```

**Diagnostic**:
- If training AUC >> validation AUC: **Increase λ** (overfitting)
- If both training and validation AUC are low: **Decrease λ** (underfitting)

---

## Full Hyperparameter Search Example

### Grid Search with Cross-Validation

```python
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import pandas as pd

# Define parameter grid
param_grid = {
    'rho': [0.3, 0.5, 0.7],
    'latent_dim': [10, 20, 30],
    'lambda_reg': [0.001, 0.01, 0.1]
}

# Prepare data
data = EnsembleData(R, labels)
train_data, val_data = data.split_labeled_data(train_fraction=0.8, random_state=42)

# Grid search
results = []
for params in ParameterGrid(param_grid):
    trainer = CFEnsembleTrainer(
        n_classifiers=R.shape[0],
        rho=params['rho'],
        latent_dim=params['latent_dim'],
        lambda_reg=params['lambda_reg'],
        max_iter=50,
        verbose=False
    )
    
    trainer.fit(train_data)
    
    # Evaluate
    val_pred = trainer.predict(val_data)
    val_idx = val_data.labeled_idx
    auc = roc_auc_score(val_data.labels[val_idx], val_pred[val_idx])
    
    results.append({**params, 'val_auc': auc})

# Analyze results
df = pd.DataFrame(results)
df = df.sort_values('val_auc', ascending=False)

print("Top 5 configurations:")
print(df.head())

# Best configuration
best_params = df.iloc[0].to_dict()
print(f"\nBest parameters:")
print(f"  ρ = {best_params['rho']:.1f}")
print(f"  d = {best_params['latent_dim']}")
print(f"  λ = {best_params['lambda_reg']:.4f}")
print(f"  Val AUC = {best_params['val_auc']:.4f}")
```

**Time**: O(|grid| × training_time). For 3×3×3=27 configurations, ~5-10 minutes on typical datasets.

---

### Bayesian Optimization (Advanced)

For expensive evaluations, use Bayesian optimization:

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Define search space
space = [
    Real(0.0, 1.0, name='rho'),
    Integer(10, 50, name='latent_dim'),
    Real(0.001, 0.1, name='lambda_reg', prior='log-uniform')
]

@use_named_args(space)
def objective(rho, latent_dim, lambda_reg):
    trainer = CFEnsembleTrainer(
        n_classifiers=R.shape[0],
        rho=rho,
        latent_dim=latent_dim,
        lambda_reg=lambda_reg,
        max_iter=50,
        verbose=False
    )
    trainer.fit(train_data)
    
    val_pred = trainer.predict(val_data)
    val_idx = val_data.labeled_idx
    auc = roc_auc_score(val_data.labels[val_idx], val_pred[val_idx])
    
    return -auc  # Minimize negative AUC

# Run optimization
result = gp_minimize(objective, space, n_calls=20, random_state=42)

print(f"Best parameters found:")
print(f"  ρ = {result.x[0]:.2f}")
print(f"  d = {result.x[1]}")
print(f"  λ = {result.x[2]:.4f}")
print(f"  Val AUC = {-result.fun:.4f}")
```

**Advantage**: More efficient than grid search (fewer evaluations).  
**Typical runs**: 15-30 evaluations vs 27+ for grid search.

---

## Practical Guidelines

### Quick Start (Minimal Tuning)

If you need quick results without extensive tuning:

```python
trainer = CFEnsembleTrainer(
    n_classifiers=m,
    latent_dim=20,      # Works for most problems
    rho=0.5,            # Balanced default
    lambda_reg=0.01,    # Standard regularization
    max_iter=50
)
```

This configuration works well ~70-80% of the time.

---

### When to Tune Each Parameter

**Priority 1: ρ** (highest impact)
- Affects fundamental behavior (reconstruction vs supervision)
- Easy to tune (5-7 values in 0.0-1.0 range)
- **Always tune this first**

**Priority 2: λ** (regularization)
- Important for preventing overfitting
- Tune if validation performance is poor
- Use 3-5 values in log scale

**Priority 3: d** (latent dimensionality)
- Less critical if dataset is reasonably sized
- Tune if you have time/resources
- Usually 20 is sufficient

**Priority 4: Others** (aggregator learning rate, max iterations)
- Usually less impactful
- Tune only if you're stuck or need marginal improvements

---

## Adaptive ρ Strategies (Advanced)

### Time-Varying ρ

Start with reconstruction-heavy, gradually increase supervision:

```python
class AdaptiveRhoTrainer(CFEnsembleTrainer):
    def fit(self, ensemble_data, rho_schedule='linear'):
        """Train with time-varying rho."""
        # Start with high rho (reconstruction-heavy)
        rho_start = 0.8
        rho_end = 0.3
        
        for t in range(self.max_iter):
            if rho_schedule == 'linear':
                current_rho = rho_start + (rho_end - rho_start) * (t / self.max_iter)
            elif rho_schedule == 'exponential':
                current_rho = rho_start * (rho_end / rho_start) ** (t / self.max_iter)
            
            self.rho = current_rho
            # ... perform iteration ...
```

**Rationale**: Early training benefits from structure learning (high ρ), later training benefits from supervision (low ρ).

---

### Performance-Based ρ

Adjust ρ based on validation performance:

```python
# After each epoch, check validation performance
if val_auc_increased:
    # Good direction, keep rho
    pass
elif recon_loss > sup_loss:
    # Reconstruction is bottleneck, increase its weight
    rho = min(rho + 0.1, 1.0)
else:
    # Supervision is bottleneck, increase its weight
    rho = max(rho - 0.1, 0.0)
```

**Caution**: Can be unstable. Use with careful monitoring.

---

## Debugging Poor Performance

### Symptom: Training AUC is good, validation AUC is bad

**Likely cause**: Overfitting  
**Solutions**:
1. **Increase λ** (0.01 → 0.05 → 0.1)
2. **Decrease d** (30 → 20 → 10)
3. **Add more labeled data** if possible
4. **Use simpler aggregator** (mean instead of weighted)

---

### Symptom: Both training and validation AUC are bad

**Likely cause**: Underfitting or poor ρ choice  
**Solutions**:
1. **Tune ρ** (try full range 0.0-1.0)
2. **Increase d** (20 → 30 → 40)
3. **Decrease λ** (0.01 → 0.001)
4. **Check base classifiers** (are they any good individually?)

---

### Symptom: Loss plateaus early

**Likely cause**: Local minimum or ρ mismatch  
**Solutions**:
1. **Try different ρ values**
2. **Increase max_iter** (50 → 100)
3. **Different random seed** (check if it's consistent)
4. **Adjust learning rate** for aggregator

---

## Summary: Quick Decision Tree

```
Start here: ρ=0.5, d=20, λ=0.01
    ↓
How much labeled data?
    ├─ <10%: Try ρ=0.7
    ├─ 10-40%: Keep ρ=0.5
    └─ >40%: Try ρ=0.3
    ↓
Training AUC vs Val AUC?
    ├─ Both low: Try ρ=0.0 or ρ=1.0 (extremes)
    ├─ Training >> Val: Increase λ or decrease d
    └─ Both good: Done! 🎉
    ↓
Still not satisfied?
    ├─ Grid search over [ρ, λ, d]
    └─ Or try Bayesian optimization
```

---

## Experiments to Run

### Recommended Validation Experiments

1. **ρ ablation study**: Train with ρ ∈ {0.0, 0.3, 0.5, 0.7, 1.0}, plot validation AUC
2. **d sensitivity**: Train with d ∈ {10, 20, 30, 40}, check overfitting
3. **λ regularization**: Train with λ ∈ {0.001, 0.01, 0.1}, monitor train/val gap

These three experiments (~15 training runs) give excellent intuition for your specific dataset.

---

## References

- **Collaborative Filtering**: Koren et al., "Matrix Factorization Techniques for Recommender Systems" (2009)
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **Hyperparameter Optimization**: Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (2012)

---

**Recommended Reading Order**:
1. Start with **Quick Start** (minimal tuning)
2. If not satisfied, do **Method 1: Cross-Validation** (ρ tuning)
3. For production systems, do **Full Grid Search**
4. For research, explore **Adaptive ρ Strategies**

**Time investment**:
- Quick start: 5 minutes
- Basic tuning (ρ only): 30 minutes
- Full tuning (ρ, λ, d): 1-2 hours
- Advanced strategies: Ongoing research

---

*Last updated: January 2026*
