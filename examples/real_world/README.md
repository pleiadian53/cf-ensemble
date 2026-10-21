# Real-World Dataset Examples

**Status**: In progress (Phase 5: real-world validation)

CF-Ensemble on real, noisy, highly imbalanced biomedical data, run on a RunPod GPU pod via
[`ops/`](../../ops/README.md) (never locally). Scripts here are meant to mature: once stable,
the reusable parts graduate into an application layer and then `src/`.

## Realistic benchmark

Stacking vs. two-stage vs. joint CF-Ensemble. Design, hypotheses and results:
[docs/realistic_benchmark/](docs/realistic_benchmark/README.md).

| File | Role |
|---|---|
| `benchmark_data.py` | Dataset registry and loaders (`protein_homo`, `coffin_splice`) |
| `base_models.py` | Heterogeneous scikit-learn base classifiers |
| `01_generate_base_predictions.py` | Train base models on D_base, build probability matrix R on D_ens |
| `02_compare_ensembles.py` | Compare ensemble methods over label budgets and seeds |
