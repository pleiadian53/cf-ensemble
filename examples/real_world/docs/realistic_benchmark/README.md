# Realistic Benchmark: Stacking vs. Two-Stage vs. Joint CF-Ensemble

**Status:** in progress (smoke test done; COFFIN next) · **Started:** 2026-10-20
**Code:** [`examples/real_world/`](../../) · **Compute:** RunPod pod via [`ops/`](../../../../ops/README.md)

The January benchmarks used synthetic data on which simple averaging already reached
PR-AUC 1.0, so they could not show whether CF-Ensemble adds anything. This benchmark
moves to real, noisy, highly imbalanced data and asks one question:

> Does letting the labels steer the matrix factorization (joint training) beat
> factorizing first and stacking afterwards (two-stage), and does either beat plain
> stacking on the raw predictions?

## Documents

| Doc | Contents |
|---|---|
| [01_experimental_design.md](01_experimental_design.md) | Datasets, splits, methods, metrics, and the design choices behind them |
| [02_hypotheses_and_predictions.md](02_hypotheses_and_predictions.md) | When the joint objective should help, where it sits in the literature, pre-registered predictions |
| [03_results_protein_homo_smoke.md](03_results_protein_homo_smoke.md) | Smoke-test results, prediction scorecard, labeled-point distortion diagnostic, bugs fixed |

## Pipeline

```
benchmark_data.py ──▶ 01_generate_base_predictions.py ──▶ 02_compare_ensembles.py
 (load + encode)       (train ~11 base models on D_base,     (label budgets × seeds:
                        predict D_ens → R, m × n)              mean, best single, stacking,
                                                               two-stage, joint)
```

| File | Role |
|---|---|
| `benchmark_data.py` | Dataset registry: `protein_homo` (smoke test), `coffin_splice` (headline) |
| `base_models.py` | Heterogeneous scikit-learn base classifiers |
| `01_generate_base_predictions.py` | Disjoint D_base / D_ens split → probability matrix R on the volume |
| `02_compare_ensembles.py` | All ensemble methods on R; writes `results.csv`, `summary.csv` |

## Running (on the pod, never locally)

```bash
python examples/ops/ops_run_pipeline.py --execute --job-name cf-smoke -- \
  "python examples/real_world/01_generate_base_predictions.py --dataset protein_homo \
     --data-root /runpod-volume/cf-ensemble/data --out-root /runpod-volume/cf-ensemble/output && \
   python examples/real_world/02_compare_ensembles.py --dataset protein_homo \
     --out-root /runpod-volume/cf-ensemble/output"
```

Datasets are cached and results written under `/runpod-volume/cf-ensemble/` on the shared
network volume, then rsynced back to `output/`. The default pod is an RTX 4000 Ada
(8 vCPU, about $0.28/hr); see `ops/configs/gpu_config.yaml`.
