# Optimization Examples

Examples demonstrating different optimization approaches for CF-Ensemble learning.

---

## Examples in This Directory

### 1. [`compare_als_pytorch.py`](compare_als_pytorch.py) ⚖️

**Comparison** of ALS (closed-form) vs PyTorch (gradient descent) optimization.

**What it does**:
- Generates synthetic probability matrix
- Optimizes using both ALS and PyTorch
- Compares convergence speed, final loss, factor correlations
- Generates side-by-side comparison plots

**Usage**:
```bash
python examples/optimization/compare_als_pytorch.py

# Custom output
python examples/optimization/compare_als_pytorch.py --output-dir results/als_pytorch_comparison
```

**Time**: ~20 seconds

**Output**:
- Convergence curves (loss vs iteration)
- Factor correlation analysis
- Side-by-side comparison table
- `als_vs_pytorch_comparison.png`

**Key insights**:
- ALS: No learning rate, faster convergence, CPU-friendly
- PyTorch: GPU-accelerated, extensible to neural variants
- Both converge to similar solutions (mathematically equivalent)

**Related docs**: [`docs/methods/als_vs_pytorch.md`](../../docs/methods/als_vs_pytorch.md), [`docs/methods/als_mathematical_derivation.md`](../../docs/methods/als_mathematical_derivation.md)

---

## Future Examples (Planned)

### `als_basics.py`
Introduction to ALS optimization:
- Step-by-step ALS updates
- Visualization of factor matrices evolving
- Convergence analysis

### `convergence_analysis.py`
Detailed convergence study:
- Effect of initialization
- Regularization impact
- Convergence speed vs problem size

### `hyperparameter_tuning.py`
Systematic hyperparameter optimization:
- Grid search for ρ, λ, d
- Cross-validation setup
- Bayesian optimization
- Visualize hyperparameter interactions

---

## Key Concepts

### ALS (Alternating Least Squares)

**Advantages**:
- ✅ No learning rate to tune
- ✅ Guaranteed convergence
- ✅ Closed-form updates (fast per-iteration)
- ✅ Well-established for matrix factorization

**When to use**:
- Small to medium datasets (< 100K instances)
- CPU-only environments
- When stability is critical

### PyTorch Gradient Descent

**Advantages**:
- ✅ GPU acceleration (10-100x faster on large data)
- ✅ Extensible to neural variants
- ✅ Easy integration with deep learning pipelines

**When to use**:
- Large datasets (> 100K instances)
- GPU available
- Planning neural extensions

---

## Learning Path

1. **Understand ALS theory**: Read [`docs/methods/als_mathematical_derivation.md`](../../docs/methods/als_mathematical_derivation.md) (~60 min)
2. **See it in action**: Run `compare_als_pytorch.py` (~20 sec)
3. **Understand equivalence**: Compare convergence curves and final solutions
4. **Deep dive**: Read source code in `src/cfensemble/optimization/`

---

##Related Documentation

| Topic | Document | Time |
|-------|----------|------|
| ALS Derivation | [`als_mathematical_derivation.md`](../../docs/methods/als_mathematical_derivation.md) | 60 min |
| ALS vs PyTorch | [`als_vs_pytorch.md`](../../docs/methods/als_vs_pytorch.md) | 30 min |
| Hyperparameter Tuning | [`hyperparameter_tuning.md`](../../docs/methods/hyperparameter_tuning.md) | 45 min |

---

**Phase**: 2 (ALS Optimization)  
**Status**: Complete ✅  
**Dependencies**: `src/cfensemble/optimization/als.py`, `src/cfensemble/optimization/pytorch_gd.py`
