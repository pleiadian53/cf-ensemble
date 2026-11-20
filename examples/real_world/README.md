# Real-World Dataset Examples

**Status**: In progress: real-world validation

CF-Ensemble on real, noisy, highly imbalanced biomedical data, run on a RunPod GPU pod via
[`ops/`](../../ops/README.md) (never locally). Scripts here are meant to mature: once stable,
the reusable parts graduate into an application layer and then `src/`.

## Realistic benchmark

Does transforming the prediction matrix with collaborative filtering help, and how should the labels enter? Every round's question, protocol and results, and the standing verdict, are in **[the results ledger](docs/results/index.md)**.

```
benchmark_data.py ──▶ 01_generate_base_predictions.py ──▶ 02_compare_ensembles.py, 03_…, 06_…
 (load + encode)       (train 11 base models on D_base,     (label budgets × seeds: the
                        predict D_ens → R, m × n)             combiners compared on R)
```

| File | Role |
|---|---|
| `benchmark_data.py` | Dataset registry and loaders (`protein_homo`, `coffin_splice`) |
| `base_models.py` | Heterogeneous scikit-learn base classifiers |
| `01_generate_base_predictions.py` | Train base models on D_base, build the probability matrix R on D_ens |
| `02_compare_ensembles.py` | Mean, best single, stacking, ensemble selection, two-stage and joint CF ([rounds 1–2](docs/results/round-02-coffin-splice.md)) |
| `03_confidence_and_link.py` | Confidence × link × combiner on R̂, two-stage CF ([rounds 3–4](docs/results/round-04-confirmation.md)) |
| `04_joint_simplex.py` | Convex combiner trained inside CF, two-stage vs. joint ([round 5](docs/results/round-05-joint-simplex.md)) |
| `05_ranking_and_alternating.py`, `alternating.py` | Ranking-loss combiners; alternating CF and greedy selection ([round 6](docs/results/round-06-ranking-and-alternating.md)) |
| `06_label_supervision.py`, `label_supervision.py` | Label-supervised CF arms that keep the combiner's inputs label-blind, transductive and inductive (planned for round 7) |
| `fold_in.py` | An instance's latent vector from its predictions alone, for either link; used for inductive prediction and episode training |
| `confidence.py` | Confidence matrices: uniform, per-classifier, learned cell-level |
| `combiners.py` | Logistic regression, ensemble selection, SUMMA |

## Running (on the pod, never locally)

```bash
python examples/ops/ops_run_pipeline.py --execute --job-name cf-smoke -- \
  "python examples/real_world/01_generate_base_predictions.py --dataset protein_homo \
     --data-root /runpod-volume/cf-ensemble/data --out-root /runpod-volume/cf-ensemble/output && \
   python examples/real_world/02_compare_ensembles.py --dataset protein_homo \
     --out-root /runpod-volume/cf-ensemble/output"
```

Datasets are cached and results written under `/runpod-volume/cf-ensemble/` on the shared network volume, then synced back to `output/`. Pod types are configured in `ops/configs/gpu_config.yaml`; the benchmark runs so far used an A40.
