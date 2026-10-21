"""
Step 2: Compare ensemble methods on the probability matrix R from step 1.

For each label budget (fraction of D_ens whose labels are revealed) and seed, the
labeled set L and test set T are drawn (stratified) from D_ens. Every method sees
R for all of D_ens (transductive) but labels only for L; metrics are on T.

Baselines:
- mean           average of base predictions (no labels)
- summa          label-free spectral combiner on rank covariance (no labels)
- best_single    base model with the best PR-AUC on L
- ens_select     Caruana greedy ensemble selection on L
- stacking       logistic regression on raw R[:, L]

CF methods (same PyTorch factorization model and optimizer; g = logistic regression),
swept over latent dim d and confidence weighting (certainty |r−0.5| or uniform):
- two_stage      factorize with ρ=1 (labels unused), then LR on R̂[:, L]
- joint          ρ·L_recon + (1−ρ)·L_sup trained jointly, per ρ
- joint_sg       same, but supervision does not update labeled y_i (stop-gradient)
- joint_sum      original summed reconstruction loss (diagnostic; first d only)

Usage (on the pod):
    python examples/real_world/02_compare_ensembles.py --dataset protein_homo \
        --out-root /runpod-volume/cf-ensemble/output
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from combiners import ensemble_selection, summa  # noqa: E402
from cfensemble.data import EnsembleData  # noqa: E402
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-root", default="output")
    p.add_argument("--base-seed", type=int, default=0, help="which step-1 run to load")
    p.add_argument("--label-fracs", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--latent-dims", type=int, nargs="+", default=[2, 4, 8],
                   help="keep d < m: with d >= m the factorization can reproduce R exactly")
    p.add_argument("--confidence", nargs="+", default=["certainty", "uniform"],
                   choices=["certainty", "uniform"])
    p.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--sg-rhos", type=float, nargs="+", default=[0.1, 0.5],
                   help="ρ values for the stop-gradient variant")
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-delta", type=float, default=1e-6)
    return p.parse_args()


def fit_lr(features_L: np.ndarray, y_L: np.ndarray, features_T: np.ndarray) -> np.ndarray:
    """Logistic-regression combiner; class-balanced like the CF trainers' supervised loss."""
    lr = LogisticRegression(class_weight="balanced", max_iter=2000)
    lr.fit(features_L, y_L)
    return lr.predict_proba(features_T)[:, 1]


def make_data(R: np.ndarray, labels: np.ndarray, confidence: str) -> EnsembleData:
    return EnsembleData(R, labels, C=np.ones_like(R) if confidence == "uniform" else None)


def make_trainer(args, m: int, d: int, rho: float, seed: int, normalize: bool = True,
                 sup_updates_instances: bool = True) -> CFEnsemblePyTorchTrainer:
    return CFEnsemblePyTorchTrainer(
        n_classifiers=m, latent_dim=d, rho=rho, lambda_reg=args.lambda_reg,
        max_epochs=args.max_epochs, lr=args.lr, patience=args.patience, min_delta=args.min_delta,
        normalize_recon=normalize, sup_updates_instances=sup_updates_instances,
        verbose=False, random_seed=seed,
    )


def main() -> None:
    args = parse_args()
    base_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.base_seed}"
    out_dir = base_dir.parent / f"compare_base{args.base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(base_dir / "R_ens.npz")
    R, y = data["R"].astype(np.float64), data["y"].astype(int)
    names = list(data["names"])
    m, n = R.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[data] R {m} × {n:,}, {y.mean():.3%} positive; models: {', '.join(names)}; device {device}")
    if max(args.latent_dims) >= m:
        print(f"[warn] latent dims {args.latent_dims} include d >= m={m}: no rank bottleneck there")

    summa_scores = summa(R)  # label-free: computed once on all of D_ens
    rows = []

    def record(method, frac, seed, scores, T, seconds, d=None, confidence=None, rho=None,
               epochs=None):
        rows.append({
            "method": method, "d": d, "confidence": confidence, "rho": rho,
            "label_frac": frac, "seed": seed,
            "pr_auc": average_precision_score(y[T], scores),
            "roc_auc": roc_auc_score(y[T], scores),
            "seconds": round(seconds, 2), "epochs": epochs,
        })

    for seed in args.seeds:
        # two_stage factorizations ignore labels: fit once per (seed, d, confidence), labels hidden
        recon = {}
        for d in args.latent_dims:
            for conf in args.confidence:
                t0 = time.time()
                tr = make_trainer(args, m, d, rho=1.0, seed=seed)
                tr.fit(make_data(R, np.full(n, np.nan), conf))
                recon[d, conf] = (tr.model.reconstruct().detach().cpu().numpy(),
                                  time.time() - t0, tr.n_epochs_)

        for frac in args.label_fracs:
            split = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
            L, T = next(split.split(np.zeros(n), y))
            labels = np.full(n, np.nan)
            labels[L] = y[L]

            record("mean", frac, seed, R[:, T].mean(axis=0), T, 0.0)
            record("summa", frac, seed, summa_scores[T], T, 0.0)
            best = int(np.argmax([average_precision_score(y[L], R[u, L]) for u in range(m)]))
            record("best_single", frac, seed, R[best, T], T, 0.0)
            t0 = time.time()
            weights = ensemble_selection(R[:, L], y[L])
            record("ens_select", frac, seed, weights @ R[:, T], T, time.time() - t0)
            t0 = time.time()
            record("stacking", frac, seed, fit_lr(R[:, L].T, y[L], R[:, T].T), T, time.time() - t0)

            for d in args.latent_dims:
                for conf in args.confidence:
                    R_hat, recon_s, recon_epochs = recon[d, conf]
                    t0 = time.time()
                    scores = fit_lr(R_hat[:, L].T, y[L], R_hat[:, T].T)
                    record("two_stage", frac, seed, scores, T, recon_s + time.time() - t0,
                           d, conf, 1.0, recon_epochs)

                    variants = [("joint", rho, True, True) for rho in args.rhos]
                    variants += [("joint_sg", rho, True, False) for rho in args.sg_rhos]
                    if d == args.latent_dims[0] and conf == args.confidence[0]:
                        variants.append(("joint_sum", 0.5, False, True))
                    for method, rho, normalize, sup_inst in variants:
                        t0 = time.time()
                        tr = make_trainer(args, m, d, rho, seed, normalize, sup_inst)
                        tr.fit(make_data(R, labels, conf))
                        record(method, frac, seed, tr.predict()[T], T, time.time() - t0,
                               d, conf, rho, tr.n_epochs_)

            done = [r for r in rows if r["seed"] == seed and r["label_frac"] == frac]
            best_cf = max((r for r in done if r["d"] is not None), key=lambda r: r["pr_auc"])
            base = {r["method"]: r["pr_auc"] for r in done if r["d"] is None}
            print(f"[seed {seed} | {frac:.0%} labeled, {int(y[L].sum())} pos] "
                  + "  ".join(f"{k}={v:.4f}" for k, v in base.items())
                  + f"  | best CF: {best_cf['method']} d={best_cf['d']} {best_cf['confidence']} "
                    f"ρ={best_cf['rho']} = {best_cf['pr_auc']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    keys = ["label_frac", "method", "d", "confidence", "rho"]
    summary = (results.groupby(keys, dropna=False)[["pr_auc", "roc_auc"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out_dir / "summary.csv")
    print("\nPR-AUC on T, mean over seeds (CF rows: best d/confidence/ρ per method):")
    cell = results.groupby(keys, dropna=False)["pr_auc"].mean().reset_index()
    best = cell.loc[cell.groupby(["label_frac", "method"])["pr_auc"].idxmax()]
    print(best.pivot(index="method", columns="label_frac", values="pr_auc").round(4))
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
