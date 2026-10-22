"""
Step 3: Can confidence weighting and a logistic link make the CF representation useful?

COFFIN showed Caruana ensemble selection winning because it drops useless base models,
while a low-rank R̂ mixes every model into every reconstructed row. This experiment
gives the CF factorization a way to do the same, and separates "is R̂ a better
representation?" from "which combiner is best?":

- confidence C:  uniform | classifier (per-model quality on L) | learned (cell-level,
                 class-balanced, cross-fitted reliability model) — see confidence.py
- link:          identity  R̂ = XᵀY
                 logit     R̂ = σ(XᵀY + b_u + b_i), cross-entropy reconstruction
- combiner on R̂: logistic regression (stacking on R̂) | ensemble selection on R̂

Factorizations are two-stage (ρ=1: labels never enter the reconstruction target);
classifier/learned confidence uses labels on L only to weight entries. Raw-R
baselines (mean, best single, stacking, ensemble selection) are rerun on the same
splits so every comparison is paired.

Usage (on the pod):
    python examples/real_world/03_confidence_and_link.py --dataset coffin_splice \
        --out-root /runpod-volume/cf-ensemble/output
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from combiners import ensemble_selection, fit_lr  # noqa: E402
from confidence import CONFIDENCE_KINDS, make_confidence  # noqa: E402
from cfensemble.data import EnsembleData  # noqa: E402
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-root", default="output")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the output dir")
    p.add_argument("--label-fracs", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--latent-dims", type=int, nargs="+", default=[4, 8])
    p.add_argument("--links", nargs="+", default=["identity", "logit"], choices=["identity", "logit"])
    p.add_argument("--confidence", nargs="+", default=["uniform", "classifier", "learned"],
                   choices=CONFIDENCE_KINDS)
    p.add_argument("--configs", nargs="+", default=None, metavar="CONF:LINK:D",
                   help="exact configurations to run (overrides the confidence × link × d grid)")
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-delta", type=float, default=1e-7)
    return p.parse_args()


def factorize(args, R: np.ndarray, C: np.ndarray, d: int, link: str, seed: int):
    """Two-stage factorization (ρ=1, no labels in the target); returns (R̂, epochs)."""
    tr = CFEnsemblePyTorchTrainer(
        n_classifiers=R.shape[0], latent_dim=d, rho=1.0, lambda_reg=args.lambda_reg,
        max_epochs=args.max_epochs, lr=args.lr, patience=args.patience, min_delta=args.min_delta,
        normalize_recon=True, link=link, verbose=False, random_seed=seed,
    )
    tr.fit(EnsembleData(R, np.full(R.shape[1], np.nan), C=C))
    return tr.model.reconstruct().detach().cpu().numpy(), tr.n_epochs_


def main() -> None:
    args = parse_args()
    base_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.base_seed}"
    out_dir = base_dir.parent / f"confidence_link_base{args.base_seed}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(base_dir / "R_ens.npz")
    R, y = data["R"].astype(np.float64), data["y"].astype(int)
    m, n = R.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[data] R {m} × {n:,}, {y.mean():.3%} positive; device {device}")

    rows = []

    def record(method, frac, seed, scores, T, seconds, **cfg):
        rows.append({"method": method, "label_frac": frac, "seed": seed,
                     "pr_auc": average_precision_score(y[T], scores),
                     "roc_auc": roc_auc_score(y[T], scores), "seconds": round(seconds, 2), **cfg})

    if args.configs:
        configs = [(c, l, int(d)) for c, l, d in (s.split(":") for s in args.configs)]
    else:
        configs = list(itertools.product(args.confidence, args.links, args.latent_dims))
    print(f"[configs] {configs}")

    for seed, frac in itertools.product(args.seeds, args.label_fracs):
        split = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
        L, T = next(split.split(np.zeros(n), y))

        # Raw-R baselines on the same split
        record("mean", frac, seed, R[:, T].mean(axis=0), T, 0.0)
        best = int(np.argmax([average_precision_score(y[L], R[u, L]) for u in range(m)]))
        record("best_single", frac, seed, R[best, T], T, 0.0)
        record("stacking", frac, seed, fit_lr(R[:, L].T, y[L], R[:, T].T), T, 0.0)
        record("ens_select", frac, seed, ensemble_selection(R[:, L], y[L]) @ R[:, T], T, 0.0)

        cache, conf_cache = {}, {}  # per split: learned reliability, C per kind
        for conf, link, d in configs:
            t0 = time.time()
            if conf not in conf_cache:
                conf_cache[conf] = make_confidence(conf, R, y, L, seed, cache)
            C = conf_cache[conf]
            conf_s = time.time() - t0

            t0 = time.time()
            R_hat, epochs = factorize(args, R, C, d, link, seed)
            fit_s = time.time() - t0 + conf_s
            cfg = dict(confidence=conf, link=link, d=d, epochs=epochs)
            record("cf_lr", frac, seed, fit_lr(R_hat[:, L].T, y[L], R_hat[:, T].T), T, fit_s, **cfg)
            weights = ensemble_selection(R_hat[:, L], y[L])
            record("cf_ens_select", frac, seed, weights @ R_hat[:, T], T, fit_s, **cfg)

        done = pd.DataFrame([r for r in rows if r["seed"] == seed and r["label_frac"] == frac])
        base = done[done.confidence.isna()] if "confidence" in done else done
        cf = done.dropna(subset=["confidence"]).sort_values("pr_auc", ascending=False).iloc[0]
        print(f"[seed {seed} | {frac:.0%} labeled, {int(y[L].sum())} pos] "
              + "  ".join(f"{r.method}={r.pr_auc:.4f}" for r in base.itertuples())
              + f"  | best CF: {cf.method} {cf.confidence} {cf.link} d={int(cf.d)} = {cf.pr_auc:.4f}",
              flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    keys = ["label_frac", "method", "confidence", "link", "d"]
    results.groupby(keys, dropna=False)[["pr_auc", "roc_auc"]].agg(["mean", "std"]).round(4) \
        .to_csv(out_dir / "summary.csv")
    print("\nPR-AUC on T, mean over seeds (fixed configurations):")
    shown = results.fillna({"confidence": "-", "link": "-", "d": 0})  # keep raw-R baselines
    piv = shown.pivot_table(index=["method", "confidence", "link", "d"], columns="label_frac",
                            values="pr_auc", aggfunc="mean")
    piv["avg"] = piv.mean(axis=1)
    print(piv.sort_values("avg", ascending=False).round(4).to_string())
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
