"""
Step 6: Can the labels shape the factorization without a train/test mismatch?

Round 2 found that letting labels into the factorization failed twice over: without
stop-gradient the labeled columns bend toward their labels; with it the joint g was
under-trained. Round 4 confirmed a two-stage method instead (rho = 1, logit link, labels
only through the confidence C, ensemble selection on R_hat). This experiment rebuilds the
label-supervised variants on the confirmed ingredients and adds inductive arms, so each
can be compared with the confirmed method on the same splits (see label_supervision.py):

  arm              setting       read-outs
  two_stage        transductive  ens_select on R_hat (the confirmed method)
  two_stage_ind    inductive     ens_select on R_hat (test columns folded in)
  matched_joint    transductive  g (logistic, stop-gradient, refit) | ens_select on R_hat
  label_row        transductive  label row | ens_select on R_hat   (label-aware: the mismatch arm)
  label_row_sg     transductive  label row | ens_select on R_hat   (all vectors label-blind)
  episodes         transductive  label row | ens_select on R_hat
  label_row_ind    inductive     label row | ens_select on R_hat
  label_row_sg_ind inductive     label row | ens_select on R_hat
  episodes_ind     inductive     label row | ens_select on R_hat

Raw-R baselines (mean, stacking, ens_select) run on the same splits. Inductive arms never
see the test columns until prediction, and each test instance is then scored on its own.
Only y[L] is read. For the label-row arms, PR-AUC on L from the free (label-aware) vectors
is also recorded: a large L-over-T gap is the symptom of a train/test mismatch.

Usage:
    python examples/real_world/06_label_supervision.py --dataset protein_homo \
        --configs learned:logit:4 --seeds 0 --label-fracs 0.05          # local smoke
    python examples/real_world/06_label_supervision.py --dataset coffin_splice \
        --configs classifier:logit:8 --out-root /runpod-volume/cf-ensemble/output   # pod
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
from label_supervision import (  # noqa: E402
    FitConfig, R_hat_from_net, factors_of, label_row, matched_joint, two_stage,
)

LABEL_ROW_ARMS = ["label_row", "label_row_sg", "episodes",
                  "label_row_ind", "label_row_sg_ind", "episodes_ind"]
ARMS = ["two_stage", "two_stage_ind", "matched_joint"] + LABEL_ROW_ARMS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-root", default="output")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the output dir")
    p.add_argument("--label-fracs", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--configs", nargs="+", required=True, metavar="CONF:LINK:D",
                   help=f"confidence ({'|'.join(CONFIDENCE_KINDS)}) : link : latent dim")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--hide-frac", type=float, default=0.2)
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--refine-epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    base_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.base_seed}"
    out_dir = base_dir.parent / f"label_supervision_base{args.base_seed}{'_' + args.tag if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(base_dir / "R_ens.npz")
    R, y = data["R"].astype(np.float64), data["y"].astype(int)
    m, n = R.shape
    print(f"[data] R {m} × {n:,}, {y.mean():.3%} positive; device {device}; arms {args.arms}")

    rows = []

    def record(method, frac, seed, scores, T, seconds, scores_L=None, L=None, **cfg):
        row = {"method": method, "label_frac": frac, "seed": seed,
               "pr_auc": average_precision_score(y[T], scores),
               "roc_auc": roc_auc_score(y[T], scores), "seconds": round(seconds, 2), **cfg}
        if scores_L is not None:
            row["pr_auc_L"] = average_precision_score(y[L], scores_L)
        rows.append(row)

    configs = [(c, l, int(d)) for c, l, d in (s.split(":") for s in args.configs)]
    for seed, frac in itertools.product(args.seeds, args.label_fracs):
        split = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
        L, T = next(split.split(np.zeros(n), y))

        record("mean", frac, seed, R[:, T].mean(axis=0), T, 0.0)
        record("stacking", frac, seed, fit_lr(R[:, L].T, y[L], R[:, T].T), T, 0.0)
        record("ens_select", frac, seed, ensemble_selection(R[:, L], y[L]) @ R[:, T], T, 0.0)

        cache = {}
        for conf, link, d in configs:
            C = make_confidence(conf, R, y, L, seed, cache)
            cfg = FitConfig(d=d, link=link, lambda_reg=args.lambda_reg, lr=args.lr,
                            max_epochs=args.max_epochs, patience=args.patience,
                            min_delta=args.min_delta, rho=args.rho, hide_frac=args.hide_frac,
                            refine_epochs=args.refine_epochs, device=device)
            tags = dict(confidence=conf, link=link, d=d)

            def sel(R_hat):  # ensemble selection on R_hat, weights chosen on L
                return ensemble_selection(R_hat[:, L], y[L]) @ R_hat[:, T]

            warm = {}
            for inductive in (False, True):
                arm = "two_stage_ind" if inductive else "two_stage"
                needs = {arm} | {a for a in LABEL_ROW_ARMS if a.endswith("_ind") == inductive}
                if not needs & set(args.arms):
                    continue
                t0 = time.time()
                cols = np.sort(L) if inductive else np.arange(n)
                net = two_stage(R, C, cols, cfg, seed)
                warm[inductive] = (factors_of(net), time.time() - t0)
                if arm in args.arms:
                    R_hat = R_hat_from_net(net, R, C, L, T, cfg, inductive)
                    record(f"{arm}:ens_select", frac, seed, sel(R_hat), T, time.time() - t0, **tags)

            if "matched_joint" in args.arms:
                t0 = time.time()
                g_T, R_hat = matched_joint(R, C, y, L, T, cfg, seed)
                s = time.time() - t0
                record("matched_joint:g", frac, seed, g_T, T, s, rho=args.rho, **tags)
                record("matched_joint:ens_select", frac, seed, sel(R_hat), T, s, rho=args.rho, **tags)

            for arm in LABEL_ROW_ARMS:
                if arm not in args.arms:
                    continue
                inductive = arm.endswith("_ind")
                f, warm_s = warm[inductive]
                t0 = time.time()
                scores_T, R_hat, scores_L, epochs = label_row(
                    R, C, y, L, T, cfg, seed, episodes=arm.startswith("episodes"),
                    inductive=inductive, warm=f,
                    blind="_sg" in arm or arm.startswith("episodes"))
                s = time.time() - t0 + warm_s
                extra = dict(rho=args.rho, refine_epochs_run=epochs, **tags)
                if arm.startswith("episodes"):
                    extra["hide_frac"] = args.hide_frac
                record(f"{arm}:label_row", frac, seed, scores_T, T, s, scores_L=scores_L, L=L, **extra)
                record(f"{arm}:ens_select", frac, seed, sel(R_hat), T, s, **extra)

        done = pd.DataFrame([r for r in rows if r["seed"] == seed and r["label_frac"] == frac])
        print(f"[seed {seed} | {frac:.0%} labeled, {int(y[L].sum())} pos] "
              + "  ".join(f"{r.method}={r.pr_auc:.4f}" for r in done.itertuples()), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    keys = [k for k in ["label_frac", "method", "confidence", "link", "d"] if k in results]
    results.groupby(keys, dropna=False)[["pr_auc", "roc_auc"]].agg(["mean", "std"]).round(4) \
        .to_csv(out_dir / "summary.csv")
    piv = results.fillna({"confidence": "-"}).pivot_table(
        index=["method", "confidence"], columns="label_frac", values="pr_auc", aggfunc="mean")
    piv["avg"] = piv.mean(axis=1)
    print("\nPR-AUC on T, mean over seeds:")
    print(piv.sort_values("avg", ascending=False).round(4).to_string())
    print(f"\n[done] -> {out_dir}")


if __name__ == "__main__":
    main()
