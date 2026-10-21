"""
Step 1: Train heterogeneous base classifiers and build the probability matrix R.

Splits the dataset (stratified) into two disjoint parts:
- D_base: trains the base classifiers
- D_ens:  the ensemble-level data; R[u, i] = classifier u's P(y=1) for instance i in D_ens

Keeping D_base and D_ens disjoint means R carries honest out-of-sample errors —
the noise CF-Ensemble is supposed to denoise — and no label leaks between levels.

Usage (on the pod; see ops/README.md):
    python examples/real_world/01_generate_base_predictions.py --dataset protein_homo \
        --data-root /runpod-volume/cf-ensemble/data --out-root /runpod-volume/cf-ensemble/output

Output: <out-root>/benchmarks/<dataset>/base_seed<seed>/
    R_ens.npz         R (m × n_ens, float32), y (n_ens,), names (m,)
    base_scores.csv   per-classifier fit time and PR-/ROC-AUC on D_ens (reference only)
    meta.json         split sizes, positive rates, settings
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from base_models import make_base_models  # noqa: E402
from benchmark_data import LOADERS, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-root", default="output")
    p.add_argument("--max-rows", type=int, default=None, help="stratified subsample before splitting")
    p.add_argument("--base-frac", type=float, default=0.5, help="fraction of data used to train base models")
    p.add_argument("--models", nargs="*", default=None, help="subset of base models (default: all)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.dataset, args.data_root, args.max_rows, args.seed)
    X_base, X_ens, y_base, y_ens = train_test_split(
        X, y, train_size=args.base_frac, stratify=y, random_state=args.seed
    )
    print(f"[data] {args.dataset}: {X.shape[0]:,} rows × {X.shape[1]} features, "
          f"{y.mean():.3%} positive")
    print(f"[data] D_base {len(y_base):,} ({y_base.sum():,} pos) | "
          f"D_ens {len(y_ens):,} ({y_ens.sum():,} pos)")

    models = make_base_models(seed=args.seed)
    if args.models:
        models = {k: models[k] for k in args.models}

    rows, names, R = [], [], []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_base, y_base)
        fit_s = time.time() - t0
        p = model.predict_proba(X_ens)[:, 1].astype(np.float32)
        ap, auc = average_precision_score(y_ens, p), roc_auc_score(y_ens, p)
        print(f"[base] {name:14s} fit {fit_s:7.1f}s  PR-AUC {ap:.4f}  ROC-AUC {auc:.4f}")
        rows.append({"model": name, "fit_seconds": round(fit_s, 2), "pr_auc": ap, "roc_auc": auc})
        names.append(name)
        R.append(np.clip(p, 0.0, 1.0))

    R = np.vstack(R)
    np.savez_compressed(out_dir / "R_ens.npz", R=R, y=y_ens.astype(np.int8), names=np.array(names))
    pd.DataFrame(rows).to_csv(out_dir / "base_scores.csv", index=False)
    meta = {
        "dataset": args.dataset, "seed": args.seed, "base_frac": args.base_frac,
        "max_rows": args.max_rows, "n_features": int(X.shape[1]),
        "n_base": int(len(y_base)), "n_ens": int(len(y_ens)),
        "pos_rate_base": float(y_base.mean()), "pos_rate_ens": float(y_ens.mean()),
        "models": names, "sklearn": sklearn.__version__, "python": platform.python_version(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] R {R.shape} -> {out_dir}")


if __name__ == "__main__":
    main()
