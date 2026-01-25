# CF-Ensemble Failure Modes

This directory documents common failure modes, pitfalls, and how to avoid them when implementing and using CF-Ensemble.

---

## Purpose

CF-Ensemble is a sophisticated method combining collaborative filtering and ensemble learning. While powerful, it has several subtle failure modes that can cause:
- Complete performance breakdown (worse than simple averaging)
- Non-convergence of optimization
- Misleading results

These documents help you:
1. **Recognize** when something is wrong
2. **Diagnose** the root cause
3. **Fix** the issue with proven solutions

---

## Failure Modes

### 1. [Transductive vs. Inductive Learning](transductive_vs_inductive.md)

**Problem:** Using traditional train/test split breaks CF-Ensemble  
**Symptom:** Performance worse than simple averaging, worse than random  
**Cause:** Treating test instances as "new" when they should be "seen"  
**Fix:** Train on ALL data with masked test labels (transductive learning)

**Critical:** This is the **#1 most common mistake**. If your CF-Ensemble performs terribly, check this first!

**Key insight from recommender systems:**
- Test instances are like "movies in your database with some ratings hidden"
- NOT like "movies you've never heard of"
- Use warm-start (learned factors), not cold-start (recompute factors)

---

### 2. [Optimization Instability](optimization_instability.md)

**Problem:** Alternating ALS + gradient descent doesn't converge  
**Symptom:** Flat supervised loss, no improvement over iterations  
**Cause:** ALS and gradient descent optimize different objectives that conflict  
**Fix:** Use joint gradient descent via PyTorch/JAX

**When this happens:**
- Reconstruction loss decreases
- Supervised loss stays flat (~0.5)
- PR-AUC stuck near random
- Never converges even after 200+ iterations

**Solutions:**
1. **Recommended:** Joint PyTorch optimization (`CFEnsemblePyTorchTrainer`)
2. **Quick fix:** Two-stage training (pure reconstruction → train aggregator)
3. **Workaround:** Damped alternating updates (slow aggregator learning)

---

## Diagnostic Checklist

If CF-Ensemble isn't working, check these in order:

### 1. Data Split ✓

- [ ] Are you training on ALL data (train + test)?
- [ ] Are test labels masked with `np.nan`?
- [ ] Are you using transductive prediction (`predict()` not `predict(R_new=...)`)?

**If NO to any:** See [transductive_vs_inductive.md](transductive_vs_inductive.md)

### 2. Convergence ✓

- [ ] Does training converge within 100-200 iterations?
- [ ] Is supervised loss decreasing?
- [ ] Is performance improving over iterations?

**If NO to any:** See [optimization_instability.md](optimization_instability.md)

### 3. Performance ✓

- [ ] Is CF-Ensemble better than simple averaging?
- [ ] Is it competitive with stacking?
- [ ] Does it improve with more labeled data?

**If NO:** Check:
- Hyperparameters (`latent_dim`, `lambda_reg`, `rho`)
- Confidence weights (label-aware vs. certainty-based)
- Data quality (are base model predictions reasonable?)

### 4. Hyperparameters ✓

- [ ] Is `latent_dim` appropriate for your data? (10-50, or ~√m)
- [ ] Is `lambda_reg` not too strong? (try 0.001-0.1)
- [ ] Is `rho` in a reasonable range? (0.3-0.7)

---

## Quick Reference: Symptoms → Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| PR-AUC < Simple Average | Wrong train/test split | Use transductive learning |
| PR-AUC ≈ Random | Wrong train/test split OR no convergence | Check data split AND convergence |
| Never converges | Optimization instability | Use PyTorch trainer |
| Supervised loss flat | Optimization instability | Use PyTorch trainer |
| Works on easy data, fails on hard | Hyperparameters | Tune `latent_dim`, `lambda_reg` |
| Predictions all similar | Over-regularization | Decrease `lambda_reg` |
| Overfitting to train | Under-regularization | Increase `lambda_reg` |

---

## Best Practices

### 1. Always Use Transductive Learning (Unless You Can't)

```python
# ✓ CORRECT: Transductive
R_all = np.hstack([R_train, R_test])
labels_all = np.concatenate([y_train, np.full(len(y_test), np.nan)])
trainer.fit(EnsembleData(R_all, labels_all))
y_pred = trainer.predict()[len(y_train):]  # Use learned factors

# ✗ WRONG: Inductive (unless truly necessary)
trainer.fit(EnsembleData(R_train, y_train))
y_pred = trainer.predict(R_new=R_test)  # Cold-start, loses information
```

### 2. Use PyTorch for Production

```python
# Recommended for production
from cfensemble.optimization import CFEnsemblePyTorchTrainer

trainer = CFEnsemblePyTorchTrainer(
    n_classifiers=m,
    latent_dim=20,
    rho=0.5,
    max_epochs=200,
    optimizer='adam',
    patience=20
)
```

**Why:**
- Guaranteed convergence
- Modern optimizers (Adam, learning rate scheduling)
- GPU acceleration
- Easier to extend

### 3. Start Simple, Then Improve

**Stage 1:** Validate the approach
```python
# Two-stage training (simple, fast)
trainer_recon = CFEnsembleTrainer(rho=1.0)  # Pure reconstruction
trainer_recon.fit(data)
# Then train aggregator separately
```

**Stage 2:** Optimize performance
```python
# Joint PyTorch optimization (better results)
trainer = CFEnsemblePyTorchTrainer(...)
trainer.fit(data)
```

### 4. Always Check Baselines First

Before trusting CF-Ensemble results:
```python
# Simple average
y_pred_simple = np.mean(R_test, axis=0)

# Stacking
from sklearn.linear_model import LogisticRegression
stacker = LogisticRegression().fit(R_train.T, y_train)
y_pred_stack = stacker.predict_proba(R_test.T)[:, 1]

# CF-Ensemble should beat simple average
# And be competitive with stacking
```

---

## Related Documentation

- [Theory](../methods/cf_ensemble_optimization_objective_tutorial.md) - Mathematical foundations
- [Examples](../examples_README.md) - Code examples and benchmarks

---

## Contributing

Found a new failure mode? Please document it:

1. **Describe the problem** - What goes wrong?
2. **Show symptoms** - How do you recognize it?
3. **Explain the cause** - Why does it happen?
4. **Provide solution** - How to fix it?
5. **Add examples** - Code snippets showing wrong vs. right

Follow the format in existing documents. PRs welcome!

---

## Lessons Learned

### From Amazon Recommender Systems

**Warm start vs. cold start:**
- Movies in database (some ratings hidden) → warm start
- Brand new movies → cold start
- CF-Ensemble is (usually) warm start!

### From Machine Learning

**Not all ML is inductive:**
- Inductive: Learn from train, apply to unseen test
- Transductive: Have test inputs (not labels) during training
- CF-Ensemble is transductive by design

### From Optimization Theory

**Alternating optimization is fragile:**
- Works when all steps optimize the SAME objective
- Fails when objectives conflict
- Joint optimization with unified gradients is more robust

---

**Remember:** Most CF-Ensemble failures are NOT bugs, but **misunderstandings of the method's assumptions**. Understanding these failure modes will save you hours of debugging!
