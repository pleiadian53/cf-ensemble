# Benchmark Examples

Experimental validation, diagnostic tools, and comparative studies.

---

## Purpose

1. **Validate CF-Ensemble** approach vs baselines
2. **Compare optimization methods** (ALS vs PyTorch)
3. **Diagnose issues** and understand failure modes
4. **Test fixes** and improvements
5. **Generate publication-ready** results

---

## Current Status (2026-01-25)

### ✅ Major Milestone: Class-Weighted Gradients Fix

**Problem Solved:** Aggregator weight collapse on imbalanced data  
**Solution:** Class-weighted gradients (inverse frequency weighting)  
**Impact:** Perfect performance restored (PR-AUC 1.000, was 0.071)

All current benchmarks reflect this critical fix.

---

## Scripts in This Directory

### 🔬 Core Diagnostic & Testing Scripts (Current)

These are **actively maintained** and reflect the latest fixes:

#### 1. **`test_class_weighted_fix.py`** ⭐ NEW
**Purpose:** Comprehensive test of class-weighted gradients fix  
**Tests:**
- ALS with/without class weighting
- PyTorch with/without class weighting
- Performance on imbalanced data (10% positive)

**Usage:**
```bash
python examples/benchmarks/test_class_weighted_fix.py
```

**Expected output:**
```
ALS (class weighted):     PR-AUC = 1.000 ✅
PyTorch (class weighted): PR-AUC = 1.000 ✅
```

---

#### 2. **`analyze_class_weighted_results.py`** ⭐ NEW
**Purpose:** Detailed analysis of class-weighted training  
**Analyzes:**
- Weight evolution and stability
- Prediction variance and quality
- ALS vs PyTorch weight diversity

**Usage:**
```bash
python examples/benchmarks/analyze_class_weighted_results.py
```

---

#### 3. **`test_pytorch_vs_als.py`** ⭐ UPDATED
**Purpose:** Compare ALS and PyTorch trainers  
**Key finding:** Both need class weighting on imbalanced data

**Usage:**
```bash
python examples/benchmarks/test_pytorch_vs_als.py
```

---

#### 4. **`synthetic_data_generator.py`** ✅ PRODUCTION-READY
**Purpose:** Generate controlled synthetic data for testing  
**Features:**
- Adjustable target quality (PR-AUC)
- Configurable imbalance ratios
- Reproducible with random seed
- **Fixed:** Now reliably achieves target quality

**Usage:**
```bash
from cfensemble.data import generate_imbalanced_ensemble_data

R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(
    n_instances=500,
    n_classifiers=10,
    n_labeled=250,
    positive_rate=0.10,  # 10% positive (imbalanced)
    target_quality=0.70,  # PR-AUC target
    random_state=42
)
```

---

### 🔍 Historical Diagnostic Scripts (Reference)

These scripts were used to **diagnose the weight collapse problem**. Kept for reference but not needed for regular use:

#### 5. **`analyze_aggregator_gradients.py`** 📊 DIAGNOSTIC
**Purpose:** Trace aggregator gradient evolution during training  
**Usage:** Understanding why weights collapsed with imbalanced data  
**Status:** Problem solved, script kept for educational purposes

---

#### 6. **`diagnose_pytorch_gradients.py`** 📊 DIAGNOSTIC  
**Purpose:** Verify PyTorch has same gradient problem as ALS  
**Key finding:** Proved problem is class imbalance, not alternating optimization

---

#### 7. **`deep_diagnostic_cf_ensemble.py`** 📊 DIAGNOSTIC
**Purpose:** Step-by-step trace of CF-Ensemble training  
**Usage:** Identified aggregator as failure point (not reconstruction)

---

#### 8. **`diagnose_synthetic_data.py`** 📊 DIAGNOSTIC
**Purpose:** Verify synthetic data generation achieves target quality  
**Status:** Issue fixed in `src/cfensemble/data/synthetic.py`

---

#### 9. **`ultra_diagnostic.py`** 📊 DIAGNOSTIC
**Purpose:** Test individual components in isolation  
**Usage:** Verified reconstruction and matrix factorization work correctly

---

### 🏗️ Older/Obsolete Scripts (Needs Review)

These scripts may need updates or replacement:

#### 10. **`cf_ensemble_benchmark.py`** ⚠️ NEEDS UPDATE
**Purpose:** Original benchmark comparing CF-Ensemble to baselines  
**Status:** Pre-dates class-weighted fix, results may not reflect current performance

---

#### 11. **`cf_ensemble_benchmark_fixed.py`** ⚠️ NEEDS REVIEW
**Purpose:** Updated benchmark after synthetic data fix  
**Status:** May predate class-weighted gradients fix

---

#### 12. **`pytorch_vs_als_benchmark.py`** ⚠️ OUTDATED
**Purpose:** Original ALS vs PyTorch comparison  
**Status:** Replaced by `test_pytorch_vs_als.py` which includes class weighting

---

#### 13. **`cf_diagnostic.py`** ⚠️ NEEDS REVIEW
**Purpose:** General diagnostic tool  
**Status:** May be redundant with newer diagnostic scripts

---

#### 14. **`test_aggregator.py`** ⚠️ NEEDS REVIEW
**Purpose:** Unit test for aggregator component  
**Status:** Should verify class-weighted gradients are tested

---

## Recommended Workflow

### For Development & Testing

1. **Generate data:**
```python
from cfensemble.data import generate_imbalanced_ensemble_data
R, labels, labeled_idx, y_true = generate_imbalanced_ensemble_data(...)
```

2. **Test class-weighted fix:**
```bash
python examples/benchmarks/test_class_weighted_fix.py
```

3. **Compare ALS vs PyTorch:**
```bash
python examples/benchmarks/test_pytorch_vs_als.py
```

### For Research & Analysis

1. **Analyze weight diversity:**
```bash
python examples/benchmarks/analyze_class_weighted_results.py
```

2. **Understand gradient dynamics:**
```bash
python examples/benchmarks/analyze_aggregator_gradients.py
```

### For Understanding Historical Issues

Review diagnostic scripts to understand:
- Why weights collapsed (class imbalance)
- How we discovered it (gradient tracing)
- Why freeze didn't fully fix it (masked the problem)
- Why PyTorch also failed (same root cause)

---

## Key Findings Summary

### 1. Class-Weighted Gradients (2026-01-25)

**Problem:** Weights collapsed to negative values on imbalanced data  
**Cause:** Majority class dominates standard gradients  
**Solution:** Inverse frequency weighting  
**Result:** Perfect performance restored

**Evidence:**
- `test_class_weighted_fix.py`: ALS & PyTorch both achieve PR-AUC = 1.000
- `analyze_class_weighted_results.py`: Weights remain positive (0.072-0.335)
- `test_pytorch_vs_als.py`: PyTorch learns 8.5x more diverse weights

### 2. Synthetic Data Quality (2026-01-25)

**Problem:** Generator not achieving target quality (78% gap)  
**Fix:** Calibrated noise levels, latent score model  
**Result:** Reliable target quality achievement

**Evidence:**
- `diagnose_synthetic_data.py`: Verified fix works across imbalance ratios

### 3. ALS vs PyTorch (2026-01-25)

**Finding:** Both need class weighting, but PyTorch learns richer weights  
**ALS:** Weight std = 0.005 (less diverse)  
**PyTorch:** Weight std = 0.041 (8.5x more diverse)

**Implication:** PyTorch preferred for accuracy, ALS for speed

---

## Planned Future Work

### Phase 4: Experimental Validation

**Still TODO** (from original roadmap):

1. **Baseline comparison** (Phase 4.2)
   - CF-Ensemble vs Simple/Weighted Avg vs Stacking
   - Update to use class-weighted trainers

2. **Rho ablation study** (Phase 4.3)
   - Systematic study of ρ ∈ {0.0, 0.1, ..., 1.0}
   - Effect on reconstruction vs supervision trade-off

3. **Label efficiency analysis** (Phase 4.4)
   - Performance vs % labeled data
   - Transductive learning advantage

4. **Real-world datasets**
   - Medical/biomedical classification
   - Comparison to published baselines

---

## Testing on Imbalanced Data

**Critical:** Always test with various imbalance ratios:

```python
for positive_rate in [0.01, 0.05, 0.10, 0.30, 0.50]:
    # Test both with and without class weighting
    test_trainer(positive_rate, use_class_weights=True)   # Should work
    test_trainer(positive_rate, use_class_weights=False)  # Should fail if imbalanced
```

**Expected:**
- With class weighting: Excellent performance at all ratios
- Without class weighting: Catastrophic failure at <20% positive

---

## Documentation Links

| Topic | Document |
|-------|----------|
| **Class-Weighted Gradients** | [`docs/methods/optimization/class_weighted_gradients.md`](../../docs/methods/optimization/class_weighted_gradients.md) |
| **Aggregator Collapse Failure Mode** | [`docs/failure_modes/aggregator_weight_collapse.md`](../../docs/failure_modes/aggregator_weight_collapse.md) |
| **ALS Mathematical Derivation** | [`docs/methods/als_mathematical_derivation.md`](../../docs/methods/als_mathematical_derivation.md) |
| **ALS vs PyTorch Comparison** | [`docs/methods/als_vs_pytorch.md`](../../docs/methods/als_vs_pytorch.md) |
| **Session Notes (2026-01-25)** | [`dev/sessions/2026-01-25_class_weighted_fix_SUCCESS.md`](../../dev/sessions/2026-01-25_class_weighted_fix_SUCCESS.md) |

---

## Script Status Summary

| Script | Status | Purpose |
|--------|--------|---------|
| `test_class_weighted_fix.py` | ⭐ **Current** | Test class weighting fix |
| `analyze_class_weighted_results.py` | ⭐ **Current** | Analyze fix results |
| `test_pytorch_vs_als.py` | ⭐ **Updated** | Compare optimization methods |
| `synthetic_data_generator.py` | ✅ **Production** | Generate test data |
| `analyze_aggregator_gradients.py` | 📊 **Diagnostic** | Gradient analysis (reference) |
| `diagnose_pytorch_gradients.py` | 📊 **Diagnostic** | PyTorch gradients (reference) |
| `deep_diagnostic_cf_ensemble.py` | 📊 **Diagnostic** | Training trace (reference) |
| `diagnose_synthetic_data.py` | 📊 **Diagnostic** | Data quality (reference) |
| `ultra_diagnostic.py` | 📊 **Diagnostic** | Component testing (reference) |
| `cf_ensemble_benchmark.py` | ⚠️ **Needs Update** | Pre-fix benchmark |
| `cf_ensemble_benchmark_fixed.py` | ⚠️ **Needs Review** | Post-data-fix benchmark |
| `pytorch_vs_als_benchmark.py` | ⚠️ **Outdated** | Replaced by test_pytorch_vs_als.py |
| `cf_diagnostic.py` | ⚠️ **Needs Review** | General diagnostic |
| `test_aggregator.py` | ⚠️ **Needs Review** | Aggregator unit test |

---

**Status:** Documentation Complete  
**Last Updated:** 2026-01-25  
**Next:** Phase 4 experimental validation with updated trainers
