# cfensemble-ops — remote-training infrastructure

Cloud infra for running cf-ensemble experiments (base-classifier training, CF-Ensemble
fits, benchmarks) on a GPU pod instead of the laptop. Adapted from `ssl-lab/ops`, and
**deliberately decoupled** from the library: nothing here imports `cfensemble`, and
`cfensemble` never imports this.

**SkyPilot orchestrates; RunPod is the provider** — `cloud: runpod` is a config
value, not a hard-coded API client, so switching providers is a config change.

## Install

```bash
pip install -e ops/            # compute-check + dry-run work with this alone
pip install -e 'ops/[remote]'  # add SkyPilot when you actually launch pods
```

## Workflow

```
compute check ──▶ provision pod ──▶ stage data ──▶ run job(s) ──▶ fetch results
```

```bash
# 1. What am I on? Do I need a pod?
python examples/ops/ops_compute_check.py

# 2. Dry-run a job (prints SkyPilot YAML + cost, no cloud calls)
python examples/ops/ops_run_pipeline.py --gpu a40 -- \
    python examples/benchmarks/pytorch_vs_als_benchmark.py

# 3. Launch for real — the pod is KEPT ALIVE by default (reuse it across sessions).
#    Results rsync back to output/; re-run more jobs on the same pod by passing
#    --cluster cf-pytorch-vs-als-benchmark (the job name).
python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- \
    python examples/benchmarks/pytorch_vs_als_benchmark.py

# 4. Stage large inputs once; they persist on the volume across pods
python examples/ops/ops_stage_data.py data/<dataset>

# 5. When done with all sessions, release the pod (either way):
python examples/ops/ops_run_pipeline.py --execute --teardown --gpu a40 -- <last job>
#   or directly:  sky down cf-pytorch-vs-als-benchmark -y
```

> Defaults: `ops_run_pipeline.py` is **dry-run** until `--execute`, and **keeps
> the pod alive** after a run (pass `--teardown` to release it) so you can run
> multiple jobs on one pod. `ops_provision_cluster.py` is the opposite — it
> **provisions for real** by default (use `--dry-run` to preview).

## Storage

The pod mounts the shared **"AI lab extension"** RunPod network volume (500 GB,
CA-MTL-1) at `/runpod-volume`. cf-ensemble keeps everything under
`/runpod-volume/cf-ensemble/` (`data/`, `output/`, `pip-cache/`) so it never
collides with sibling projects on the same volume.

For **large datasets shared across projects**, keep a single physical copy in a
project-neutral data lake and link only what you use:

```bash
export CFENSEMBLE_DATA_ROOT=~/work/_datalake
python -c "from ops.datasets import link_dataset; link_dataset('genomics/splice_sites/celegans_splice')"
# -> data/celegans_splice symlinks into $CFENSEMBLE_DATA_ROOT/genomics/splice_sites/celegans_splice
```

## Layout

```
ops/
  ops/gpu_runner.py    InfraConfig · build_skypilot_config · print_dry_run · launch · provision/status/down
  ops/hardware.py      detect_hardware · print_report (compute check)
  ops/datasets.py      data_root · link_dataset (shared data lake)
  configs/
    gpu_config.yaml    infra defaults (GPU, cloud, volume, setup, output paths)
    skypilot/generated/  auto-written SkyPilot task YAMLs (gitignored)
  tests/
examples/ops/          thin drivers: compute_check · run_pipeline · provision_cluster · stage_data
```

## Configure

Edit [configs/gpu_config.yaml](configs/gpu_config.yaml) for defaults, or override
per-run with `--gpu`, `--use-volume`, etc. GPU choices and indicative hourly
rates live in `GPU_SPECS` in [ops/gpu_runner.py](ops/gpu_runner.py) — verify
rates against live RunPod pricing before relying on cost estimates.

Note: training heterogeneous scikit-learn base classifiers is CPU-bound, so the
pod's vCPU count matters as much as the GPU for that stage.
