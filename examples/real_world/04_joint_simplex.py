"""
Step 4: Build the selection bias into CF itself — a jointly trained simplex combiner.

Round 3 showed CF reaching parity with ensemble selection only when ensemble selection
was applied *after* the factorization. Here the combiner is part of the model:
g(r̂_i) = Σ_u α_u r̂_ui with α = softmax(θ) (nonnegative, sums to 1), trained with the
factorization. The two optimizer failures diagnosed in round 2 are addressed:
- stop-gradient: supervision updates X, b_u and g, never the labeled y_i / b_i
- g gets its own learning rate, and a final refit on the frozen reconstruction

Methods per confidence × link × d (exploratory; dev seeds):
- cf_ens_select       two-stage factorization, post-hoc ensemble selection on R̂ (round 3)
- two_stage_simplex   ρ=1 factorization, then the simplex g fitted on R̂ (refit only)
- joint_simplex_sg    ρ<1: supervision shapes X, b_u and α jointly
- joint_lr_sg         same with the logistic g (does the optimizer fix rescue it?)
Raw-R ens_select and stacking run on the same splits.

Usage (on the pod):
    python examples/real_world/04_joint_simplex.py --dataset coffin_splice \
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
from confidence import make_confidence  # noqa: E402
from cfensemble.data import EnsembleData  # noqa: E402
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-root", default="output")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--label-fracs", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--configs", nargs="+", default=["classifier:logit:8", "classifier_shrunk:logit:8"],
                   metavar="CONF:LINK:D")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.9])
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-g", type=float, default=0.05)
    p.add_argument("--refit-g-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-delta", type=float, default=1e-7)
    return p.parse_args()


def make_trainer(args, m, d, link, rho, aggregator, seed, sup_updates_instances=False):
    return CFEnsemblePyTorchTrainer(
        n_classifiers=m, latent_dim=d, rho=rho, lambda_reg=args.lambda_reg,
        aggregator_type=aggregator, max_epochs=args.max_epochs, lr=args.lr, lr_g=args.lr_g,
        refit_g_steps=args.refit_g_steps, patience=args.patience, min_delta=args.min_delta,
        normalize_recon=True, sup_updates_instances=sup_updates_instances, link=link,
        verbose=False, random_seed=seed,
    )


def main() -> None:
    args = parse_args()
    base_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.base_seed}"
    out_dir = base_dir.parent / f"joint_simplex_base{args.base_seed}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(base_dir / "R_ens.npz")
    R, y = data["R"].astype(np.float64), data["y"].astype(int)
    names = list(data["names"])
    m, n = R.shape
    configs = [(c, l, int(d)) for c, l, d in (s.split(":") for s in args.configs)]
    print(f"[data] R {m} × {n:,}, {y.mean():.3%} positive; device "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}; configs {configs}", flush=True)

    rows = []

    def record(method, frac, seed, scores, T, seconds, **cfg):
        rows.append({"method": method, "label_frac": frac, "seed": seed,
                     "pr_auc": average_precision_score(y[T], scores),
                     "roc_auc": roc_auc_score(y[T], scores), "seconds": round(seconds, 2), **cfg})

    for seed, frac in itertools.product(args.seeds, args.label_fracs):
        split = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
        L, T = next(split.split(np.zeros(n), y))
        labels = np.full(n, np.nan)
        labels[L] = y[L]

        record("stacking", frac, seed, fit_lr(R[:, L].T, y[L], R[:, T].T), T, 0.0)
        record("ens_select", frac, seed, ensemble_selection(R[:, L], y[L]) @ R[:, T], T, 0.0)

        cache, conf_cache = {}, {}
        for conf, link, d in configs:
            if conf not in conf_cache:
                conf_cache[conf] = make_confidence(conf, R, y, L, seed, cache)
            data_L = EnsembleData(R, labels, C=conf_cache[conf])
            cfg = dict(confidence=conf, link=link, d=d)

            # Two-stage: ρ=1 (labels never touch the factorization), then g on R̂
            t0 = time.time()
            tr = make_trainer(args, m, d, link, 1.0, "simplex", seed).fit(data_L)
            R_hat = tr.model.reconstruct().detach().cpu().numpy()
            fit_s = time.time() - t0
            record("cf_ens_select", frac, seed, ensemble_selection(R_hat[:, L], y[L]) @ R_hat[:, T],
                   T, fit_s, rho=1.0, **cfg)
            record("two_stage_simplex", frac, seed, tr.predict()[T], T, fit_s, rho=1.0, **cfg)

            for rho in args.rhos:
                t0 = time.time()
                tr = make_trainer(args, m, d, link, rho, "simplex", seed).fit(data_L)
                alpha = tr.model.simplex_weights().detach().cpu().numpy()
                record("joint_simplex_sg", frac, seed, tr.predict()[T], T, time.time() - t0, rho=rho,
                       epochs=tr.n_epochs_, top_models=",".join(names[i] for i in np.argsort(-alpha)[:3]),
                       **cfg)
            t0 = time.time()
            tr = make_trainer(args, m, d, link, max(args.rhos), "weighted", seed).fit(data_L)
            record("joint_lr_sg", frac, seed, tr.predict()[T], T, time.time() - t0,
                   rho=max(args.rhos), epochs=tr.n_epochs_, **cfg)

        done = pd.DataFrame([r for r in rows if r["seed"] == seed and r["label_frac"] == frac])
        summary = done.groupby("method").pr_auc.max().sort_values(ascending=False)
        print(f"[seed {seed} | {frac:.0%} labeled, {int(y[L].sum())} pos] "
              + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    shown = results.fillna({"confidence": "-", "link": "-", "d": 0, "rho": 0})
    piv = shown.pivot_table(index=["method", "confidence", "link", "d", "rho"], columns="label_frac",
                            values="pr_auc", aggfunc="mean")
    piv["avg"] = piv.mean(axis=1)
    print("\nPR-AUC on T, mean over seeds (fixed configurations):")
    print(piv.sort_values("avg", ascending=False).round(4).to_string())
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
