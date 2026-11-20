"""
Step 5: Ranking objectives for selection inside CF (exploratory; dev seeds).

Round 5 showed the in-model simplex combiner fails because cross-entropy on averaged
probabilities rewards extreme-probability models. Here the supervised term is a pairwise
AUC surrogate instead, and greedy selection is kept in the loop:

- cf_ens_select            two-stage factorization, post-hoc ensemble selection (reference)
- two_stage_simplex_rank   ρ=1 factorization, simplex g fitted with the ranking loss
- joint_simplex_rank       simplex g + factorization trained jointly with the ranking loss
- joint_lr_rank            same with the logistic g (lambda_g = 1/(2|L|))
- alt_select               alternating greedy selection / ranking-steered factorization
Raw-R ens_select and stacking run on the same splits.

Usage (on the pod):
    python examples/real_world/05_ranking_and_alternating.py --dataset coffin_splice \
        --out-root /runpod-volume/cf-ensemble/output
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from alternating import alternating_cf_selection  # noqa: E402
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
    p.add_argument("--configs", nargs="+", default=["classifier:logit:8"], metavar="CONF:LINK:D")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.9])
    p.add_argument("--pairwise-scale", type=float, default=20.0)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs-per-round", type=int, default=200)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--refit-g-steps", type=int, default=500)
    return p.parse_args()


def make_trainer(args, m, d, link, rho, aggregator, seed, lambda_g=0.0, scale=None):
    return CFEnsemblePyTorchTrainer(
        n_classifiers=m, latent_dim=d, rho=rho, lambda_reg=0.01, aggregator_type=aggregator,
        max_epochs=args.max_epochs, lr=0.01, lr_g=0.05, refit_g_steps=args.refit_g_steps,
        patience=50, min_delta=1e-7, normalize_recon=True, sup_updates_instances=False,
        lambda_g=lambda_g, link=link, sup_loss="pairwise",
        pairwise_scale=args.pairwise_scale if scale is None else scale,
        verbose=False, random_seed=seed,
    )


def main() -> None:
    args = parse_args()
    base_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.base_seed}"
    out_dir = base_dir.parent / f"ranking_alt_base{args.base_seed}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(base_dir / "R_ens.npz")
    R, y = data["R"].astype(np.float64), data["y"].astype(int)
    m, n = R.shape
    configs = [(c, l, int(d)) for c, l, d in (s.split(":") for s in args.configs)]
    print(f"[data] R {m} × {n:,}, {y.mean():.3%} positive; configs {configs}", flush=True)
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

        cache = {}
        for conf, link, d in configs:
            C = make_confidence(conf, R, y, L, seed, cache)
            data_L = EnsembleData(R, labels, C=C)
            cfg = dict(confidence=conf, link=link, d=d)

            t0 = time.time()
            tr = make_trainer(args, m, d, link, 1.0, "simplex", seed).fit(data_L)
            R_hat = tr.model.reconstruct().detach().cpu().numpy()
            s = time.time() - t0
            record("cf_ens_select", frac, seed, ensemble_selection(R_hat[:, L], y[L]) @ R_hat[:, T],
                   T, s, rho=1.0, **cfg)
            record("two_stage_simplex_rank", frac, seed, tr.predict()[T], T, s, rho=1.0, **cfg)

            for rho in args.rhos:
                t0 = time.time()
                tr = make_trainer(args, m, d, link, rho, "simplex", seed).fit(data_L)
                record("joint_simplex_rank", frac, seed, tr.predict()[T], T, time.time() - t0,
                       rho=rho, **cfg)
                t0 = time.time()
                alpha, R_hat, _ = alternating_cf_selection(
                    R, y, L, C, d, link, rho, seed, rounds=args.rounds,
                    epochs_per_round=args.epochs_per_round, pairwise_scale=args.pairwise_scale,
                    init_epochs=args.max_epochs)
                record("alt_select", frac, seed, alpha @ R_hat[:, T], T, time.time() - t0,
                       rho=rho, **cfg)

            t0 = time.time()
            tr = make_trainer(args, m, d, link, max(args.rhos), "weighted", seed,
                              lambda_g=1.0 / (2 * len(L)), scale=1.0).fit(data_L)
            record("joint_lr_rank", frac, seed, tr.predict()[T], T, time.time() - t0,
                   rho=max(args.rhos), **cfg)

        done = pd.DataFrame([r for r in rows if r["seed"] == seed and r["label_frac"] == frac])
        best = done.groupby("method").pr_auc.max().sort_values(ascending=False)
        print(f"[seed {seed} | {frac:.0%} labeled, {int(y[L].sum())} pos] "
              + "  ".join(f"{k}={v:.4f}" for k, v in best.items()), flush=True)

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
