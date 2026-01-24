# Theory vs. Empirics: What Can Be Proven?

**Last Updated**: January 24, 2026

## Overview

This document clarifies what aspects of confidence weighting effectiveness can be **mathematically proven** versus what requires **empirical validation**.

---

## Summary Table

| Question | Can Prove? | Evidence Type | Status |
|----------|-----------|---------------|--------|
| Does confidence weighting help? | ❌ No | Empirical | ✅ Verified (+1.7%) |
| Below some threshold, it doesn't help | ✅ Yes* | Theoretical + Empirical | 🔄 Theory done, validating threshold |
| Above some threshold, minimal gains | ✅ Yes | Theoretical (ceiling) | 🔄 Need empirical threshold |
| Diversity is necessary | ✅ Yes | Theoretical (proof) | ✅ Proven |
| Specific threshold values (60%, 80%) | ❌ No | Empirical only | ⏳ Need experiments |
| Strategy rankings by quality | ❌ No | Empirical | ⏳ Need experiments |
| Improvement magnitudes (+3-8%) | ❌ No | Empirical | ⏳ Need experiments |

*With assumptions (see below)

---

## What Can Be Proven

### 1. Information-Theoretic Lower Bound ✅

**Claim**: If classifiers are only slightly better than random, confidence weighting cannot help significantly.

**Proof Sketch**:

Let $p_{\text{correct}}$ be the probability a classifier is correct, and suppose $p_{\text{correct}} = 0.5 + \epsilon$ for small $\epsilon$.

The **mutual information** between confidence score $c$ and correctness $y$ is:
$$I(c; y) = H(y) - H(y | c)$$

Where:
- $H(y) \approx 1$ bit (for balanced classes)
- $H(y | c)$ is entropy of $y$ given confidence $c$

When classifiers are near-random ($\epsilon \approx 0$):
- Confidence scores weakly correlate with correctness
- $I(c; y) = O(\epsilon)$ bits

**Implication**: With $\epsilon < 0.1$ (i.e., accuracy < 60%), the confidence signal is too weak to exploit effectively.

**What we CANNOT prove**: The exact threshold (60% vs 55% vs 65%).

---

### 2. Ceiling Effect ✅

**Claim**: If baseline accuracy is already $1 - \delta$, maximum possible improvement is $\delta$.

**Proof**: Trivial.

Accuracy cannot exceed 100%. If baseline is 90%, maximum possible improvement is 10 percentage points.

In practice, irreducible error (Bayes error) means actual improvement $\ll \delta$.

**Implication**: At 85%+ accuracy, gains will be small (<5% realistically, <10% theoretically).

**What we CANNOT prove**: The exact quality level where returns become negligible (85% vs 80% vs 90%).

---

### 3. Diversity Necessity ✅

**Claim**: If all classifiers are identical, no weighting strategy can improve performance.

**Proof**:

Suppose all classifiers produce the same prediction: $r_{ui} = r_i$ for all $u$.

Any weighted ensemble prediction is:
$$\hat{y}_i = g\left(\sum_{u=1}^m w_u r_i\right) = g\left(r_i \sum_{u=1}^m w_u\right)$$

Since $\sum_{u} w_u$ is constant across instances, this is equivalent to $\hat{y}_i = g(r_i)$ for some function $g$.

Thus, all weighting schemes produce the same predictions → no weighting can improve over uniform.

**Implication**: Diversity is **necessary** for confidence weighting to help.

**What we CANNOT prove**: How much diversity is sufficient, or how to quantify "enough" diversity.

---

### 4. Calibration-Strategy Interaction ✅

**Claim**: If confidence scores are **anti-calibrated** (high confidence → low accuracy), certainty-based weighting **hurts** performance.

**Proof**:

Certainty strategy: $c_{ui} = |r_{ui} - 0.5|$ (weight by distance from 0.5).

If anti-calibrated:
- High confidence ($|r_{ui} - 0.5|$ large) → Low accuracy
- Low confidence ($|r_{ui} - 0.5|$ small) → High accuracy

Certainty strategy upweights high-confidence predictions, which are **systematically wrong** under anti-calibration.

Expected performance: **Worse** than uniform weighting.

**Empirical confirmation**: In our experiments, certainty strategy achieved **-1.3%** (worse than baseline) when classifiers had calibration issues.

**Implication**: Fixed strategies can hurt if assumptions violated. Learned reliability is more robust.

---

## What CANNOT Be Proven (Requires Empirics)

### 1. Specific Threshold Values ❌

**Question**: Is the minimum viable quality 60% or 55% or 65%?

**Why unprovable**: Depends on:
- **Problem difficulty distribution**: Easy problems have lower thresholds
- **Classifier types**: Neural nets vs. decision trees have different calibration
- **Feature quality**: Better features → better confidence signals even at lower accuracy
- **Dataset properties**: Size, noise level, class imbalance

**Need**: Systematic experiments across quality levels and datasets.

**Status**: 
- ✅ Theory says "some threshold exists"
- ⏳ Experiments needed to determine actual value

---

### 2. Strategy Rankings ❌

**Question**: Which strategy is best at which quality level?

**Why unprovable**: Strategy effectiveness depends on:
- Calibration quality (varies by classifier)
- Diversity patterns (varies by ensemble)
- Label availability (affects label-aware strategies)
- Instance difficulty distribution (varies by dataset)

**Need**: Quality × Strategy × Dataset grid search.

**Status**: 
- ✅ Observed: Learned > Calibration > Certainty (at 73% quality in our data)
- ⏳ Need: Systematic variation to establish general patterns

---

### 3. Improvement Magnitudes ❌

**Question**: How much improvement should we expect? +3-8%?

**Why unprovable**: Gain depends on:
- **Exploitable patterns**: How much do classifiers differ in their reliability profiles?
- **Quality-diversity interaction**: High diversity amplifies gains at moderate quality
- **Subgroup structure**: More complex subgroups → larger potential gains

**Need**: Real-world datasets with known subgroup structures.

**Status**:
- ✅ Observed: +1.7% at 73% quality (synthetic)
- ⏳ Expected: Larger gains on real biomedical data (more complex patterns)

---

### 4. Domain Generalization ❌

**Question**: Do thresholds transfer across domains (vision → NLP → biomedical)?

**Why unprovable**: Different domains have:
- Different classifier calibration properties
- Different instance difficulty distributions  
- Different subgroup structures
- Different base classifier quality levels

**Need**: Multi-domain empirical study.

**Status**: ⏳ Not yet investigated

---

## Empirical Validation Plan

### Experiment 1: Quality Sweep ⏳

**Script**: `examples/quality_threshold_experiment.py`

**Design**:
- Vary quality: 50%, 55%, 60%, ..., 95%
- For each: Generate data, train all strategies, measure improvement
- 5 trials per quality level

**Will answer**:
- ✅ Minimum viable quality (where improvement > 1%)
- ✅ Peak improvement quality (sweet spot)
- ✅ Diminishing returns threshold

**Expected result**: Inverted-U curve peaking at 70-80% quality.

**Status**: ✅ Script ready, ⏳ need to run

---

### Experiment 2: Quality × Diversity Grid ⏳

**Design**:
```python
qualities = [0.60, 0.70, 0.80]
diversities = ['low', 'medium', 'high']
# 3 × 3 = 9 conditions
```

**Will answer**:
- ✅ Does diversity amplify gains?
- ✅ Is diversity more important at certain quality levels?

**Expected result**: Diversity effect strongest at moderate quality (70%).

**Status**: ⏳ Not yet implemented

---

### Experiment 3: Real Biomedical Datasets 🎯

**Datasets**:
1. Gene expression classification (multiple tissue types)
2. Clinical text analysis (ICD code prediction)
3. Medical image ensembles (radiology diagnosis)

**Will answer**:
- ✅ Do thresholds hold on real data?
- ✅ Are gains larger than synthetic (+5-12% hypothesized)?
- ✅ Domain-specific variations?

**Status**: ⏳ Need access to datasets

---

## Current Evidence Status

### What We Know (Verified) ✅

| Finding | Evidence | Confidence |
|---------|----------|-----------|
| Confidence weighting **can** improve | Observed +1.7% | **High** ✅ |
| Some strategies **hurt** if miscalibrated | Observed -1.3% (certainty) | **High** ✅ |
| Learned reliability > Fixed strategies | Consistent across runs | **High** ✅ |
| Diversity is necessary | Theoretical proof + observation | **Very High** ✅ |
| There exists a lower threshold | Information theory | **High** ✅ |
| There exists an upper threshold | Ceiling effect (math) | **Very High** ✅ |

### What We Think (Hypothesized) 🔄

| Hypothesis | Confidence | Next Step |
|-----------|-----------|-----------|
| 60% minimum viable quality | **Medium** | Run Experiment 1 |
| 65-80% sweet spot | **Medium** | Run Experiment 1 |
| >85% diminishing returns | **Medium-High** | Run Experiment 1 |
| Diversity amplifies gains | **Medium** | Run Experiment 2 |
| +3-8% at sweet spot | **Low-Medium** | Real data experiments |
| Larger gains on real data | **Medium** | Biomedical datasets |

### What We Don't Know ❓

- Exact threshold values for different domains
- Strategy rankings at each quality level
- Interaction with other hyperparameters (ρ, λ, d)
- Multi-class classification thresholds
- Active learning integration effects

---

## Recommendations

### For Documentation

1. **Be explicit about evidence status**:
   - ✅ Proven theoretically
   - ✅ Verified empirically
   - 🔄 Hypothesized (being validated)
   - ❓ Unknown

2. **Update claims as experiments complete**:
   - After Experiment 1: Update threshold values
   - After Experiment 2: Update diversity effects
   - After Experiment 3: Add domain-specific guidance

3. **Acknowledge limitations**:
   - Synthetic data may not reflect real-world complexity
   - Thresholds may vary by domain
   - Guidelines are starting points, not guarantees

### For Users

**Current best practice**:

1. **Before using confidence weighting**:
   ```python
   from cfensemble.utils import diagnose_ensemble_quality
   
   recommendation = diagnose_ensemble_quality(R, labels, labeled_idx)
   print_diagnosis(recommendation)
   ```

2. **Interpret recommendations as guidelines**:
   - If diagnosis says "POOR" (<60%) → Fix classifiers **likely** better than weighting
   - If diagnosis says "OPTIMAL" (60-85%) → Weighting **likely** helps
   - If diagnosis says "EXCELLENT" (>85%) → Weighting **likely** minimal impact

3. **Always validate empirically** on your data:
   - Try multiple strategies
   - Use cross-validation
   - Don't assume thresholds transfer exactly

### For Researchers

**Open questions** (publication opportunities):

1. **Theoretical**: Can we derive tighter bounds on improvement as a function of quality and diversity?

2. **Empirical**: Do thresholds generalize across domains (vision, NLP, biomedical, tabular)?

3. **Methodological**: Can we predict improvement **before** training (diagnostic tool)?

4. **Extensions**: How do thresholds change for:
   - Multi-class classification?
   - Imbalanced datasets?
   - Online/streaming data?
   - Non-IID data?

---

## Conclusion

### The 80/20 Rule

**80% is theory + informed reasoning**:
- ✅ Some lower threshold exists (proven)
- ✅ Some upper threshold exists (proven)
- ✅ Diversity is necessary (proven)
- ✅ Fixed strategies can hurt (observed)

**20% is specific numbers**:
- ⏳ 60% vs 55% vs 65% minimum (needs experiments)
- ⏳ 70-80% vs 65-75% sweet spot (needs experiments)
- ⏳ +3-8% vs +2-5% expected gain (needs experiments)

### Honest Summary

**What we can say with confidence**:
> "Confidence weighting effectiveness depends on base classifier quality. There exists a minimum quality below which it doesn't help (information-theoretic), and an upper quality above which gains are minimal (ceiling effect). Our initial experiments suggest a minimum around 60% accuracy and peak gains at 70-80%, but systematic validation is needed to confirm these specific thresholds."

**What we should NOT claim yet**:
> ~~"Confidence weighting requires 60% minimum accuracy."~~ (Too specific without validation)

**Better framing**:
> "Based on initial experiments and theory, we hypothesize a minimum viable quality around 60% accuracy. Experiment 1 will validate this threshold systematically."

---

**Status**: Living document, updated as experiments complete.  
**Next Update**: After running `quality_threshold_experiment.py`  
**Contributors**: CF-Ensemble Development Team
