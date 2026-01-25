# Confidence Weighting Examples

Examples demonstrating confidence weighting strategies and learned reliability weights (**Phase 3**).

---

## Examples in This Directory

### 1. [`phase3_confidence_weighting.py`](phase3_confidence_weighting.py) ⭐

**Comprehensive demonstration** of all confidence strategies.

**What it does**:
- Compares 5 fixed confidence strategies
- Demonstrates learned reliability weights
- Shows ROC-AUC improvements
- Includes realistic synthetic data with subgroup structure

**Usage**:
```bash
python examples/confidence_weighting/phase3_confidence_weighting.py

# Specify output directory
python examples/confidence_weighting/phase3_confidence_weighting.py --output-dir results/my_experiment
```

**Time**: ~30 seconds

**Output**: Performance comparison table + ROC-AUC metrics

**Related docs**: [`docs/methods/confidence_weighting/`](../../docs/methods/confidence_weighting/)

---

### 2. [`reliability_model_demo.py`](reliability_model_demo.py) 🎯

**Deep dive** into the reliability weight model.

**What it does**:
- Step-by-step reliability learning
- Feature importance analysis
- 6 visualizations of learned patterns
- Weight vs quality correlation

**Usage**:
```bash
python examples/confidence_weighting/reliability_model_demo.py

# Custom output
python examples/confidence_weighting/reliability_model_demo.py --output-dir results/reliability_analysis
```

**Time**: ~45 seconds

**Output**: 6 PNG figures + analysis summary

**Related docs**: [`docs/methods/confidence_weighting/polarity_models_tutorial.md`](../../docs/methods/confidence_weighting/polarity_models_tutorial.md)

---

### 3. [`quality_threshold_experiment.py`](quality_threshold_experiment.py) 🔬

**Research experiment** validating quality thresholds.

**What it does**:
- Systematically varies base classifier quality (50% to 95%)
- Tests all strategies at each quality level
- Multiple trials for statistical robustness
- Generates comprehensive analysis plots

**Usage**:
```bash
# Full experiment (~10-15 minutes)
python examples/confidence_weighting/quality_threshold_experiment.py --diversity high --trials 5

# Quick test
python examples/confidence_weighting/quality_threshold_experiment.py --trials 3
```

**Time**: ~10-15 minutes (full), ~5 minutes (quick)

**Output**:
- `results/quality_threshold/raw_results.csv`
- `results/quality_threshold/summary.csv`
- `results/quality_threshold/quality_threshold_analysis.png` (4-panel plot)

**Purpose**: Validate hypothesized thresholds (60% min, 70-80% sweet spot)

**Related docs**: [`docs/methods/confidence_weighting/base_classifier_quality_analysis.md`](../../docs/methods/confidence_weighting/base_classifier_quality_analysis.md)

---

## Learning Path

### Beginner (First time with confidence weighting)

1. Read [`docs/methods/confidence_weighting/README.md`](../../docs/methods/confidence_weighting/README.md) (10 min)
2. Run `phase3_confidence_weighting.py` to see all strategies (30 sec)
3. Read [`docs/methods/confidence_weighting/polarity_models_tutorial.md`](../../docs/methods/confidence_weighting/polarity_models_tutorial.md) (40 min)

### Intermediate (Understanding learned reliability)

1. Run `reliability_model_demo.py` to see detailed analysis (45 sec)
2. Examine generated visualizations
3. Modify `generate_sample_data()` to test with your own data characteristics

### Advanced (Research & validation)

1. Read [`docs/methods/confidence_weighting/theory_vs_empirics.md`](../../docs/methods/confidence_weighting/theory_vs_empirics.md) (15 min)
2. Run `quality_threshold_experiment.py` to validate thresholds (15 min)
3. Analyze results in `results/quality_threshold/`
4. Compare with your own datasets

---

## Key Concepts

### Fixed Strategies

**Uniform**: All predictions weighted equally (baseline)

**Certainty**: Weight by $|r_{ui} - 0.5|$ (distance from uncertain)

**Label-aware**: Reward correct predictions on labeled data

**Calibration**: Weight by calibration quality (Brier score)

**Adaptive**: Learned combination of above features

### Learned Reliability

**Key insight**: Learn cell-level weights $W_{ui}$ that predict when classifier $u$ is reliable on instance $i$.

**Advantages**:
- Discovers complex patterns (subgroup-specific performance)
- No pseudo-labels needed
- +3-8% improvement in optimal scenarios

**When it helps**:
- Base classifier quality: 60-85% (sweet spot: 70-80%)
- High diversity (different classifier strengths/weaknesses)
- Subgroup structure in data

---

## Common Workflows

### Workflow 1: Evaluate Strategies on Your Data

```python
from cfensemble.data import EnsembleData, get_confidence_strategy
from cfensemble.models import ReliabilityWeightModel
from cfensemble.optimization import CFEnsembleTrainer

# Your probability matrix and labels
R = ...  # (m, n)
labels = ...  # (n,) with NaN for unlabeled

# Test learned reliability
rel_model = ReliabilityWeightModel(model_type='gbm')
rel_model.fit(R, labels, labeled_mask, classifier_stats)
W = rel_model.predict(R, classifier_stats)

ensemble_data = EnsembleData(R, labels, C=W)
trainer = CFEnsembleTrainer(n_classifiers=m, rho=0.5)
trainer.fit(ensemble_data)
y_pred = trainer.predict()
```

### Workflow 2: Diagnose if Confidence Weighting Will Help

```python
# Use diagnostic function from quality_threshold_experiment.py
from examples.confidence_weighting.quality_threshold_experiment import diagnose_ensemble_quality

recommendation = diagnose_ensemble_quality(R, labels, labeled_idx)
print_diagnosis(recommendation)

# Output tells you:
# - Average classifier quality
# - Diversity level
# - Recommendation (OPTIMAL, POOR, EXCELLENT, LOW_DIVERSITY)
# - Expected gain
```

---

## Related Documentation

| Topic | Document | Time |
|-------|----------|------|
| Overview | [`confidence_weighting/README.md`](../../docs/methods/confidence_weighting/README.md) | 5 min |
| Quality Thresholds | [`base_classifier_quality_analysis.md`](../../docs/methods/confidence_weighting/base_classifier_quality_analysis.md) | 30 min |
| Theory vs Empirics | [`theory_vs_empirics.md`](../../docs/methods/confidence_weighting/theory_vs_empirics.md) | 15 min |
| Reliability Learning | [`polarity_models_tutorial.md`](../../docs/methods/confidence_weighting/polarity_models_tutorial.md) | 40 min |

---

## Future Examples (Planned)

- `fixed_strategies_demo.py` - Isolated demonstration of each fixed strategy
- `calibration_analysis.py` - Deep dive into calibration-based weighting
- `confidence_distributions.py` - Visualize confidence score distributions
- `subgroup_performance.py` - Analyze performance by instance subgroups

---

**Phase**: 3 (Confidence Weighting & Reliability Learning)  
**Status**: Complete ✅  
**Dependencies**: `src/cfensemble/data/confidence.py`, `src/cfensemble/models/reliability.py`
