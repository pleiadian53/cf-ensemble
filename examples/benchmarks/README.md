# Benchmark Examples

Experimental validation and ablation studies (**Phase 4**).

---

## Purpose

Systematic experiments to:
1. Validate CF-Ensemble approach vs baselines
2. Understand effect of key hyperparameters (especially ρ)
3. Analyze label efficiency (performance vs labeled data amount)
4. Generate publication-ready results

---

## Examples in This Directory

### 1. `synthetic_data_generator.py` ⏳ (Phase 4.1)

**Flexible synthetic data generation** with controlled properties.

**Features**:
- Adjustable latent structure
- Controlled noise levels
- Label-dependent biases
- Known ground truth factors

**Usage**:
```bash
python examples/benchmarks/synthetic_data_generator.py \
    --n-samples 1000 \
    --n-classifiers 10 \
    --latent-dim 5 \
    --noise-level 0.1
```

---

### 2. `baseline_comparison.py` ⏳ (Phase 4.2)

**Compare CF-Ensemble against standard ensemble methods**.

**Baselines**:
1. Simple averaging
2. Weighted averaging (learned from validation)
3. Stacking (logistic regression meta-learner)
4. CF-Ensemble (ρ=1.0, pure reconstruction)
5. **CF-Ensemble (ρ=0.5, combined objective)** ⭐

**Metrics**:
- ROC-AUC
- Accuracy, Precision, Recall, F1
- Brier score (calibration)
- Training time

**Usage**:
```bash
python examples/benchmarks/baseline_comparison.py \
    --dataset synthetic \
    --n-trials 10
```

**Expected result**: CF-Ensemble (ρ=0.5) > CF-Ensemble (ρ=1.0) > Stacking > Weighted avg > Simple avg

---

### 3. `rho_ablation_study.py` ⏳ (Phase 4.3)

**Systematic study of ρ's effect** (reconstruction vs supervision trade-off).

**Design**:
- Test ρ ∈ {0.0, 0.1, 0.2, ..., 1.0}
- Multiple datasets (varying label amounts)
- Plot performance vs ρ

**Expected findings**:
- Low ρ (0.0-0.3): Overfits to labels when few labels
- Medium ρ (0.4-0.6): Best trade-off ⭐
- High ρ (0.7-1.0): Reproduces base model errors

**Usage**:
```bash
python examples/benchmarks/rho_ablation_study.py \
    --rho-range "0.0,0.1,0.2,...,1.0" \
    --n-labeled 50,100,200,500
```

---

### 4. `label_efficiency_analysis.py` ⏳ (Phase 4.4)

**How does performance scale with labeled data?**

**Design**:
- Fix total dataset size (e.g., 1000 instances)
- Vary labeled %: 5%, 10%, 20%, 50%, 100%
- Compare methods at each labeled %

**Key question**: Does CF-Ensemble maintain advantage with few labels?

**Expected result**: Gap largest at 10-30% labeled (where transductive learning helps most)

**Usage**:
```bash
python examples/benchmarks/label_efficiency_analysis.py \
    --labeled-percentages 5,10,20,50,100 \
    --n-trials 20
```

---

## Experimental Workflow

### Standard Benchmark Run

```bash
# 1. Generate data
python examples/benchmarks/synthetic_data_generator.py \
    --output results/benchmarks/data/

# 2. Run baseline comparison
python examples/benchmarks/baseline_comparison.py \
    --data results/benchmarks/data/ \
    --output results/benchmarks/baseline_comparison/

# 3. Rho ablation
python examples/benchmarks/rho_ablation_study.py \
    --data results/benchmarks/data/ \
    --output results/benchmarks/rho_ablation/

# 4. Label efficiency
python examples/benchmarks/label_efficiency_analysis.py \
    --data results/benchmarks/data/ \
    --output results/benchmarks/label_efficiency/
```

### Publication-Ready Figures

All scripts generate:
- High-resolution PNG/PDF figures (300 dpi)
- CSV files with raw results
- Statistical significance tests
- Confidence intervals (bootstrapped)

---

## Expected Results Summary

Based on roadmap specifications:

| Experiment | Key Finding | Expected Improvement |
|------------|-------------|---------------------|
| Baseline Comparison | CF (ρ=0.5) > CF (ρ=1.0) | +3-5% over pure reconstruction |
| Baseline Comparison | CF (ρ=0.5) > Stacking | +2-4% over stacking |
| Rho Ablation | Optimal ρ ≈ 0.4-0.6 | Sweet spot identified |
| Label Efficiency | Gap largest at 10-30% labeled | +5-10% advantage in low-label regime |

---

## Learning Path

### Prerequisites

1. Complete Phase 1-3 examples
2. Understand combined optimization objective
3. Familiar with experimental design

### Recommended Order

1. **Generate data**: `synthetic_data_generator.py` (understand data properties)
2. **Baseline comparison**: See CF-Ensemble vs simple methods
3. **Rho ablation**: Understand ρ's critical role
4. **Label efficiency**: See transductive learning advantage

---

## Implementation Status

- [ ] `synthetic_data_generator.py` - Phase 4.1
- [ ] `baseline_comparison.py` - Phase 4.2
- [ ] `rho_ablation_study.py` - Phase 4.3
- [ ] `label_efficiency_analysis.py` - Phase 4.4

**Next**: Implement Phase 4.1 (synthetic data generator)

---

## Related Documentation

| Topic | Document |
|-------|----------|
| Implementation Roadmap | [`dev/planning/IMPLEMENTATION_ROADMAP.md`](../../dev/planning/IMPLEMENTATION_ROADMAP.md) |
| Hyperparameter Tuning | [`docs/methods/hyperparameter_tuning.md`](../../docs/methods/hyperparameter_tuning.md) |
| Combined Objective | [`docs/methods/cf_ensemble_optimization_objective_tutorial.md`](../../docs/methods/cf_ensemble_optimization_objective_tutorial.md) |

---

**Phase**: 4 (Experimental Validation)  
**Status**: In Progress 🔄  
**Target**: Complete benchmarking framework for systematic evaluation
