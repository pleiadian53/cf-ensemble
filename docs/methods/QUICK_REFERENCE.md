# Quick Reference: Imbalanced Data & CF-Ensemble

**One-page cheat sheet** for working with imbalanced biomedical data.

---

## Random Baseline Performance

| Minority | Accuracy | PR-AUC | ROC-AUC | F1-Score |
|----------|----------|--------|---------|----------|
| **1%** | 0.990 ❌ | **0.010** ✅ | 0.500 ⚠️ | 0.020 |
| **5%** | 0.950 ❌ | **0.050** ✅ | 0.500 ⚠️ | 0.095 |
| **10%** | 0.900 ❌ | **0.100** ✅ | 0.500 ⚠️ | 0.182 |
| **50%** | 0.500 ✅ | **0.500** ✅ | 0.500 ✅ | 0.667 |

**Key:** 
- ✅ Use this metric
- ❌ Misleading for imbalanced data
- ⚠️ Insensitive to imbalance

**Rule of Thumb:** PR-AUC random baseline ≈ minority rate

---

## Performance Interpretation

### PR-AUC Multipliers (vs. Random)

| Multiplier | Interpretation | Clinical Value |
|------------|----------------|----------------|
| < 2x | ⚠️ Poor | Barely better than guessing |
| 2-5x | Fair | Some signal, needs improvement |
| 5-10x | **Good** | Clinically useful |
| 10-20x | **Excellent** | Strong predictive power |
| > 20x | **Outstanding** | Near-optimal |

**Example:** At 5% minority, 0.20 PR-AUC = 4x random = Fair performance

---

## Clinical Significance Thresholds

| Application | Prevalence | Min PR-AUC | Good PR-AUC | Excellent | Key Metric |
|-------------|-----------|-----------|-------------|-----------|------------|
| **Cancer screening** | 1-5% | 0.10-0.15 | 0.20-0.40 | > 0.50 | High recall |
| **Sepsis prediction** | 3-5% | 0.20-0.30 | 0.35-0.50 | > 0.60 | Catch all |
| **Rare disease** | 1-5% | 0.15-0.25 | 0.30-0.50 | > 0.60 | Target test |
| **Drug response** | 20-40% | 0.40-0.50 | 0.55-0.70 | > 0.75 | Cost-effective |
| **Splice sites** | 0.1-1% | 0.05-0.10 | 0.15-0.30 | > 0.40 | Annotation |

**Note:** Thresholds are context-dependent! Always consult domain experts.

---

## Method Selection (2026)

### Quick Decision Tree

```
Minority class rate?
│
├─ < 1% → Foundation Model + Few-Shot
│          OR Active Learning + Anomaly Detection
│
├─ 1-5% → XGBoost + Focal Loss + SMOTE
│          OR CF-Ensemble + Active Learning (if unlabeled data)
│
├─ 5-10% → CF-ENSEMBLE 🏆🏆🏆 (OPTIMAL!)
│           Expected gain: +1-4%
│
└─ 10-50% → Standard ML + Class Weights
            OR CF-Ensemble (still works!)
```

---

## CF-Ensemble Performance (Validated 2026-01-24)

| Imbalance | Random | Peak Gain | Best Baseline | Status |
|-----------|--------|-----------|---------------|--------|
| **10% pos** | 0.10 | **+1.06%** | 0.603 | ✅ Recommended |
| **5% pos** ⭐ | 0.05 | **+3.94%** 🏆 | 0.197 | ✅✅✅ **OPTIMAL** |
| **1% pos** | 0.01 | **+0.10%** | 0.030 | ❌ Skip |

**Key Finding:** 5% minority shows BEST gains (non-monotonic relationship!)

**Why 5% is optimal:**
- Not too easy (10% baseline already good)
- Just right (challenging but learnable)
- Too hard (1% fundamental limits)

---

## When to Use CF-Ensemble

### ✅✅✅ Strong Recommendation

- **Minority class:** 5-10%
- **Labeled samples:** 100-10,000
- **Unlabeled data:** Available
- **Ensemble size:** m = 5-15
- **Need interpretability:** Yes

**Expected gain:** +1-4% PR-AUC

---

### ✅ Good Candidate

- **Minority class:** 2-5% or 10-20%
- **Have diverse classifiers**
- **Limited compute budget**

**Expected gain:** +0.5-2% PR-AUC (test first!)

---

### ❌ Not Recommended

- **Minority class:** < 1%
  - Use: Foundation models, active learning
  - Why: Too few positives to learn patterns
  
- **Ensemble size:** m ≥ 15 AND baseline excellent
  - Simple averaging already near-optimal

---

## Code Snippets

### Compute Random Baselines

```python
def compute_random_baselines(minority_rate):
    return {
        'pr_auc': minority_rate,
        'roc_auc': 0.5,
        'f1': 2 * minority_rate / (1 + minority_rate),
        'accuracy': max(minority_rate, 1 - minority_rate)
    }

# Example
baselines = compute_random_baselines(0.05)
print(f"5% minority random baselines:")
print(f"  PR-AUC: {baselines['pr_auc']:.3f}")  # 0.050
print(f"  F1: {baselines['f1']:.3f}")         # 0.095
```

### Interpret Performance

```python
from sklearn.metrics import average_precision_score

pr_auc = average_precision_score(y_true, y_pred_proba)
random = minority_rate  # e.g., 0.05

multiplier = pr_auc / random
print(f"PR-AUC: {pr_auc:.3f} ({multiplier:.1f}x random)")

if multiplier < 2:
    print("⚠️ Poor: Barely better than random")
elif multiplier < 5:
    print("Fair: Some signal")
elif multiplier < 10:
    print("✅ Good: Clinically useful")
else:
    print("✅ Excellent: Strong signal")
```

### Use CF-Ensemble

```python
from cfensemble.models import ReliabilityWeightModel

# Learn confidence weights
model = ReliabilityWeightModel(n_estimators=30)
model.fit(R, labels, labeled_mask, classifier_stats)

# Weighted prediction
W = model.predict_weights(R, classifier_stats)
ensemble_pred = (R @ W) / W.sum()

# Evaluate
pr_auc = average_precision_score(y_true, ensemble_pred)
print(f"PR-AUC: {pr_auc:.3f} ({pr_auc/0.05:.1f}x random)")
```

---

## Common Pitfalls

### ❌ DON'T

1. **Use accuracy for imbalanced data**
   - 99% accuracy at 1% minority = useless!

2. **Trust ROC-AUC for severe imbalance**
   - 0.70 ROC-AUC might mean 10% precision

3. **Forget to stratify splits**
   - Test set might have 0 positives!

4. **Use threshold 0.5**
   - Predicted probabilities rarely exceed 0.5 at 1% minority

5. **Apply SMOTE before splitting**
   - Data leakage! Synthetic neighbors in test set

### ✅ DO

1. **Use PR-AUC as primary metric**
   - Focuses on minority class

2. **Report relative to random**
   - "0.20 PR-AUC (4x random)" is informative

3. **Stratify all splits**
   - `train_test_split(..., stratify=y)`

4. **Find optimal threshold**
   - Use precision-recall curve on validation

5. **SMOTE only on training**
   - Split first, augment training only

---

## State-of-the-Art (2026)

| Method | Labeled | Unlabeled | Imbalance | Compute | Interpretable |
|--------|---------|-----------|-----------|---------|---------------|
| **XGBoost + Focal** | Many | No | Good | Fast | Yes |
| **Foundation Model** | Few | Many | Excellent | Expensive | No |
| **SMOTE + Ensemble** | Moderate | No | Good | Fast | Yes |
| **CF-Ensemble** 🏆 | Moderate | Yes | Excellent (5-10%) | Fast | Yes |

**CF-Ensemble advantages:**
- ✅ Leverages unlabeled data (semi-supervised)
- ✅ Optimal at 5-10% minority (validated!)
- ✅ Interpretable confidence weights
- ✅ No synthetic data needed
- ✅ Fast training

---

## Further Reading

### Full Documentation

- **[Imbalanced Data Tutorial](imbalanced_data_tutorial.md)** - Complete guide (30 min read)
- **[When to Use Confidence Weighting](confidence_weighting/when_to_use_confidence_weighting.md)** - Decision trees
- **[Experimental Results](../RESULTS_2026-01-24.md)** - Validation details

### Code Examples

- **`examples/confidence_weighting/quality_threshold_experiment.py`** - Run experiments
- **`src/cfensemble/data/synthetic.py`** - Generate test data
- **`scripts/compare_imbalance_scenarios.py`** - Compare scenarios

---

## Key Takeaways (TL;DR)

1. **PR-AUC ≈ minority rate** for random classifier
2. **Good performance = 5-10x random** for clinical applications
3. **CF-Ensemble optimal at 5-10% minority** (+1-4% gains)
4. **< 1% minority: Use foundation models** (not CF-Ensemble)
5. **Always stratify, never use accuracy, report vs. random!**

---

**Last Updated:** 2026-01-24  
**Status:** ✅ Validated with experiments  
**For questions:** See [Imbalanced Data Tutorial](imbalanced_data_tutorial.md)
