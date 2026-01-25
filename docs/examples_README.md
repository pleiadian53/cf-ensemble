# Examples and Demonstrations

This document describes the runnable examples in the **`examples/`** directory of the CF-Ensemble repository.

**Note:** These are Python scripts in the repository, not documentation files. To run them, clone the repository and navigate to the `examples/` directory.

## 📁 Directory Structure

```
examples/
├── basics/                    # Core functionality (Phase 1-2) ⏳
├── optimization/              # ALS vs PyTorch, tuning (Phase 2) ✅
├── confidence_weighting/      # Strategies & reliability (Phase 3) ✅
├── benchmarks/                # Experimental validation (Phase 4) 🔄
├── real_world/                # Real datasets (Phase 5) ⏳
├── analysis/                  # Visualization & diagnostics (Phase 6) ⏳
└── advanced/                  # Extensions (Future) ⏳
```

**Legend**: ✅ Complete | 🔄 In Progress | ⏳ Planned

---

## 🎯 Quick Start by Goal

### I want to...

**...understand confidence weighting**  
→ `examples/confidence_weighting/` - Start with `phase3_confidence_weighting.py`

**...compare ALS vs PyTorch optimization**  
→ `examples/optimization/` - Run `compare_als_pytorch.py`

**...benchmark CF-Ensemble vs baselines**  
→ `examples/benchmarks/` - See Phase 4 scripts (in progress)

**...validate quality thresholds**  
→ `examples/confidence_weighting/` - Run `quality_threshold_experiment.py`

**...see learned reliability in action**  
→ `examples/confidence_weighting/` - Run `reliability_model_demo.py`

---

## 🔬 Research & Validation Experiments

**See:** `examples/confidence_weighting/quality_threshold_experiment.py` in the repository

---

## 📚 Examples by Phase

### Phase 2: Optimization ✅

**Directory**: `examples/optimization/`

| Script | Description | Time |
|--------|-------------|------|
| `compare_als_pytorch.py` | ALS vs PyTorch comparison | ~20s |

**Status**: Complete  
**Docs**: [ALS Mathematical Derivation](methods/als_mathematical_derivation.md)

---

### Phase 3: Confidence Weighting ✅

**Directory**: `examples/confidence_weighting/`

| Script | Description | Time |
|--------|-------------|------|
| `phase3_confidence_weighting.py` | All strategies comparison | ~30s |
| `reliability_model_demo.py` | Detailed reliability analysis | ~45s |
| `quality_threshold_experiment.py` | Systematic validation | ~10-15min |

**Status**: Complete  
**Docs**: [Confidence Weighting Methods](methods/confidence_weighting/README.md)

**Quick start**:
```bash
# Clone the repository and run:
python examples/confidence_weighting/phase3_confidence_weighting.py
```

---

### Phase 4: Benchmarks & Validation 🔄

**Directory**: `examples/benchmarks/`

| Script | Description | Status |
|--------|-------------|--------|
| `test_class_weighted_fix.py` | Class weighting validation | ✅ Complete |
| `test_pytorch_vs_als.py` | ALS vs PyTorch comparison | ✅ Complete |
| `analyze_class_weighted_results.py` | Detailed analysis | ✅ Complete |
| `synthetic_data_generator.py` | Flexible data generation | ✅ Fixed |
| `baseline_comparison.py` | vs averaging, stacking | ⏳ Planned |
| `rho_ablation_study.py` | Effect of ρ parameter | ⏳ Planned |
| `label_efficiency_analysis.py` | Performance vs labeled % | ⏳ Planned |

**Status**: Core testing complete, full validation in progress  
**See:** [Benchmarks README](https://github.com/pleiadian53/cf-ensemble/tree/main/examples/benchmarks) for all scripts

---

### Phase 5: Real-World Datasets ⏳

**Directory**: `examples/real_world/` - Planned

---

### Phase 6: Analysis & Diagnostics ⏳

**Directory**: `examples/analysis/` - Planned

---

## 📖 Documentation & Notebooks

Each example directory has its own README with:
- Detailed script descriptions
- Usage examples
- Learning paths
- Links to related documentation

**See also**:
- 📚 [Methods Documentation](methods/README.md) - Theoretical documentation
- 📓 [Jupyter Notebooks](notebooks/README.md) - Interactive tutorials

---

## 🚀 Development Workflow

**Recommended approach** (as per project organization):

1. **Develop example script** under `examples/<topic>/`
   - Pure Python, executable with `argparse`
   - Import from `src/cfensemble/`
   - Save outputs to `results/<topic>/`

2. **Test thoroughly**
   - Unit tests in `tests/`
   - Integration test via script execution

3. **Create notebook** (optional, for pedagogy)
   - under `notebooks/<topic>/`
   - Import from example script
   - Add narrative and visualizations

4. **Document** under `docs/methods/<topic>/`
   - Theoretical background
   - API documentation
   - Link to examples and notebooks

---

## 🔄 Migration Notes

**Recent reorganization** (Jan 24, 2026):
- Created topic-specific subdirectories mirroring `docs/` structure
- Moved existing scripts to appropriate locations:
  - `compare_als_pytorch.py` → `optimization/`
  - `reliability_model_demo.py` → `confidence_weighting/`
  - `phase3_confidence_weighting.py` → `confidence_weighting/`
  - `quality_threshold_experiment.py` → `confidence_weighting/`

**Recent updates** (2026-01-25):
- ✅ Fixed synthetic data generator to achieve target quality
- ✅ Discovered and fixed aggregator weight collapse (class-weighted gradients)
- ✅ Validated both ALS and PyTorch trainers on imbalanced data
- 🔄 Full validation suite in progress

See the [Benchmarks directory](https://github.com/pleiadian53/cf-ensemble/tree/main/examples/benchmarks) in the repository for all scripts and detailed README.

---

**Last Updated**: January 25, 2026  
**Status**: Phase 3 Complete ✅ | Core fixes complete ✅ | Phase 4 validation in progress 🔄
