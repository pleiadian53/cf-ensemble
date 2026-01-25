# Examples and Demonstrations

This directory contains examples and demonstrations of the **CF-Ensemble framework**, organized by topic.

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
→ [`confidence_weighting/`](confidence_weighting/) - Start with `phase3_confidence_weighting.py`

**...compare ALS vs PyTorch optimization**  
→ [`optimization/`](optimization/) - Run `compare_als_pytorch.py`

**...benchmark CF-Ensemble vs baselines**  
→ [`benchmarks/`](benchmarks/) - See Phase 4 scripts (in progress)

**...validate quality thresholds**  
→ [`confidence_weighting/`](confidence_weighting/) - Run `quality_threshold_experiment.py`

**...see learned reliability in action**  
→ [`confidence_weighting/`](confidence_weighting/) - Run `reliability_model_demo.py`

---

## 🔬 Research & Validation Experiments

**Moved to**: [`confidence_weighting/quality_threshold_experiment.py`](confidence_weighting/quality_threshold_experiment.py)

---

## 📚 Examples by Phase

### Phase 2: Optimization ✅

**Directory**: [`optimization/`](optimization/)

| Script | Description | Time |
|--------|-------------|------|
| [`compare_als_pytorch.py`](optimization/compare_als_pytorch.py) | ALS vs PyTorch comparison | ~20s |

**Status**: Complete  
**Docs**: [`docs/methods/als_mathematical_derivation.md`](../docs/methods/als_mathematical_derivation.md)

---

### Phase 3: Confidence Weighting ✅

**Directory**: [`confidence_weighting/`](confidence_weighting/)

| Script | Description | Time |
|--------|-------------|------|
| [`phase3_confidence_weighting.py`](confidence_weighting/phase3_confidence_weighting.py) | All strategies comparison | ~30s |
| [`reliability_model_demo.py`](confidence_weighting/reliability_model_demo.py) | Detailed reliability analysis | ~45s |
| [`quality_threshold_experiment.py`](confidence_weighting/quality_threshold_experiment.py) | Systematic validation | ~10-15min |

**Status**: Complete  
**Docs**: [`docs/methods/confidence_weighting/`](../docs/methods/confidence_weighting/)

**Quick start**:
```bash
# See all strategies in action
python examples/confidence_weighting/phase3_confidence_weighting.py
```

---

### Phase 4: Benchmarks & Validation 🔄

**Directory**: [`benchmarks/`](benchmarks/)

| Script | Description | Status |
|--------|-------------|--------|
| `synthetic_data_generator.py` | Flexible data generation | ⏳ Implementing |
| `baseline_comparison.py` | vs averaging, stacking | ⏳ Planned |
| `rho_ablation_study.py` | Effect of ρ parameter | ⏳ Planned |
| `label_efficiency_analysis.py` | Performance vs labeled % | ⏳ Planned |

**Status**: In Progress  
**Docs**: [`dev/planning/IMPLEMENTATION_ROADMAP.md`](../dev/planning/IMPLEMENTATION_ROADMAP.md) (Phase 4)

---

### Phase 5: Real-World Datasets ⏳

**Directory**: [`real_world/`](real_world/) - Planned

---

### Phase 6: Analysis & Diagnostics ⏳

**Directory**: [`analysis/`](analysis/) - Planned

---

## 📖 Documentation & Notebooks

Each example directory has its own README with:
- Detailed script descriptions
- Usage examples
- Learning paths
- Links to related documentation

**See also**:
- 📚 [`docs/methods/`](../docs/methods/) - Theoretical documentation
- 📓 [`notebooks/`](../notebooks/) - Interactive tutorials (to be modernized)

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

**Currently implementing**:
- `benchmarks/synthetic_data_generator.py` (Phase 4.1)
- Baseline comparison scripts
- Ablation studies

See [`benchmarks/README.md`](benchmarks/README.md) for details.

---

**Last Updated**: January 24, 2026  
**Status**: Phase 3 Complete ✅ | Phase 4 In Progress 🔄
