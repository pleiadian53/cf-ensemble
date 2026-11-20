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

Settings can be fixed (``--configs``, ``--rho``, ``--hide-frac``) or chosen on the labeled set
by cross-validation (``--select-on-L``, see selection.py): the confidence type and latent
dimension once per cell with the confirmed method, shared by every arm, then rho, and the
hidden fraction for the episode arms, per arm. The choices are recorded in results.csv and
every candidate's out-of-fold score in selection.csv.

Usage:
    python examples/real_world/06_label_supervision.py --dataset protein_homo --select-on-L \
        --seeds 0 --label-fracs 0.05                                     # local smoke, selected
    python examples/real_world/06_label_supervision.py --dataset protein_homo \
        --configs learned:logit:4 --seeds 0 --label-fracs 0.05          # local smoke
    python examples/real_world/06_label_supervision.py --dataset coffin_splice \
        --configs classifier:logit:8 --out-root /runpod-volume/cf-ensemble/output   # pod
"""

import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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
from selection import (  # noqa: E402
    confirmed_scorer, cv_select, describe, grid, joint_scorer, label_row_scorer,
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
    p.add_argument("--configs", nargs="+", default=None, metavar="CONF:LINK:D",
                   help=f"fixed configurations, confidence ({'|'.join(CONFIDENCE_KINDS)}) : link : "
                        "latent dim; required unless --select-on-L")
    p.add_argument("--select-on-L", action="store_true",
                   help="choose confidence, d, rho and hide_frac on L by cross-validation")
    p.add_argument("--link", default="logit", choices=["identity", "logit"],
                   help="the link used with --select-on-L")
    p.add_argument("--conf-grid", nargs="+", default=["uniform", "classifier", "learned"],
                   choices=CONFIDENCE_KINDS)
    p.add_argument("--d-grid", type=int, nargs="+", default=[4, 8])
    p.add_argument("--rho-grid", type=float, nargs="+", default=[0.5, 0.9])
    p.add_argument("--hide-grid", type=float, nargs="+", default=[0.1, 0.3])
    p.add_argument("--cv-folds", type=int, default=3)
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
    p.add_argument("--jobs", type=int, default=1, help="cells run in parallel processes")
    p.add_argument("--threads", type=int, default=1, help="torch threads per process")
    p.add_argument("--resume", action="store_true", help="skip cells already saved under cells/")
    args = p.parse_args()
    if not args.select_on_L and not args.configs:
        p.error("give --configs, or --select-on-L")
    return args


def run_cell(args, R: np.ndarray, y: np.ndarray, seed: int, frac: float, device: str):
    """One (seed, label budget) cell: baselines, selection on L if asked, every arm.

    Returns (rows, selection rows). Self-contained, so cells can run in separate processes.
    """
    n = R.shape[1]
    rows, sel_rows = [], []
    split = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    L, T = next(split.split(np.zeros(n), y))

    def record(method, scores, seconds, scores_L=None, **cfg):
        row = {"method": method, "label_frac": frac, "seed": seed,
               "pr_auc": average_precision_score(y[T], scores),
               "roc_auc": roc_auc_score(y[T], scores), "seconds": round(seconds, 2), **cfg}
        if scores_L is not None:
            row["pr_auc_L"] = average_precision_score(y[L], scores_L)
        rows.append(row)

    def select(name, candidates, scorer):
        """Run one cross-validated choice on L; log every candidate; return the winner."""
        t0 = time.time()
        best, table = cv_select(candidates, scorer, y, L, n_folds=args.cv_folds, seed=seed)
        for r in table.itertuples():
            sel_rows.append({"label_frac": frac, "seed": seed, "choice": name,
                             "candidate": r.candidate, "oof_pr_auc": r.oof_pr_auc, "n_folds": r.n_folds,
                             "selected": r.candidate == repr(best), "seconds": round(time.time() - t0, 1)})
        print(f"  [seed {seed} | {frac:.0%}] select {name}: {describe(table, best)} in "
              f"{time.time() - t0:.0f}s", flush=True)
        return best

    t0 = time.time()
    record("mean", R[:, T].mean(axis=0), 0.0)
    record("stacking", fit_lr(R[:, L].T, y[L], R[:, T].T), time.time() - t0)
    t0 = time.time()
    record("ens_select", ensemble_selection(R[:, L], y[L]) @ R[:, T], time.time() - t0)

    base_cfg = FitConfig(link=args.link, lambda_reg=args.lambda_reg, lr=args.lr,
                         max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta,
                         rho=args.rho, hide_frac=args.hide_frac, refine_epochs=args.refine_epochs,
                         device=device)
    cache = {}
    if args.select_on_L:
        conf, d = select("confidence_d", grid(args.conf_grid, args.d_grid),
                         confirmed_scorer(R, y, base_cfg, seed))
        configs = [(conf, args.link, d)]
    else:
        configs = [(c, l, int(d)) for c, l, d in (s.split(":") for s in args.configs)]

    for conf, link, d in configs:
        C = make_confidence(conf, R, y, L, seed, cache)
        cfg = replace(base_cfg, d=d, link=link)
        tags = dict(confidence=conf, link=link, d=d, selected_on_L=bool(args.select_on_L))

        def chosen(arm):
            """(rho, hide_frac) for an arm: selected on L, or the fixed values."""
            if not args.select_on_L:
                return args.rho, args.hide_frac
            if arm == "matched_joint":
                return select(arm, list(args.rho_grid), joint_scorer(R, y, conf, cfg, seed)), args.hide_frac
            episodes, inductive = arm.startswith("episodes"), arm.endswith("_ind")
            blind = "_sg" in arm or episodes
            scorer = label_row_scorer(R, y, conf, cfg, seed, episodes, inductive, blind)
            cands = grid(args.rho_grid, args.hide_grid) if episodes else list(args.rho_grid)
            best = select(arm, cands, scorer)
            return best if episodes else (best, args.hide_frac)

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
                record(f"{arm}:ens_select", sel(R_hat), time.time() - t0, **tags)

        if "matched_joint" in args.arms:
            rho, _ = chosen("matched_joint")
            t0 = time.time()
            g_T, R_hat = matched_joint(R, C, y, L, T, replace(cfg, rho=rho), seed)
            s = time.time() - t0
            record("matched_joint:g", g_T, s, rho=rho, **tags)
            record("matched_joint:ens_select", sel(R_hat), s, rho=rho, **tags)

        for arm in LABEL_ROW_ARMS:
            if arm not in args.arms:
                continue
            inductive = arm.endswith("_ind")
            f, warm_s = warm[inductive]
            rho, hide = chosen(arm)
            t0 = time.time()
            scores_T, R_hat, scores_L, epochs = label_row(
                R, C, y, L, T, replace(cfg, rho=rho, hide_frac=hide), seed,
                episodes=arm.startswith("episodes"), inductive=inductive, warm=f,
                blind="_sg" in arm or arm.startswith("episodes"))
            s = time.time() - t0 + warm_s
            extra = dict(rho=rho, refine_epochs_run=epochs, **tags)
            if arm.startswith("episodes"):
                extra["hide_frac"] = hide
            record(f"{arm}:label_row", scores_T, s, scores_L=scores_L, **extra)
            record(f"{arm}:ens_select", sel(R_hat), s, **extra)

    print(f"[seed {seed} | {frac:.0%} labeled, {int(y[L].sum())} pos] "
          + "  ".join(f"{r['method']}={r['pr_auc']:.4f}" for r in rows), flush=True)
    return rows, sel_rows


def _worker_init(threads: int) -> None:
    torch.set_num_threads(threads)


def _cell_paths(cells_dir: Path, seed: int, frac: float):
    stem = f"seed{seed}_frac{frac:g}"
    return cells_dir / f"{stem}_results.csv", cells_dir / f"{stem}_selection.csv"


def main() -> None:
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    base_dir = Path(args.out_root) / "benchmarks" / args.dataset / f"base_seed{args.base_seed}"
    out_dir = base_dir.parent / f"label_supervision_base{args.base_seed}{'_' + args.tag if args.tag else ''}"
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(base_dir / "R_ens.npz")
    R, y = data["R"].astype(np.float64), data["y"].astype(int)
    m, n = R.shape
    print(f"[data] R {m} × {n:,}, {y.mean():.3%} positive; device {device}; arms {args.arms}; "
          f"{args.jobs} job(s) × {args.threads} thread(s)")

    cells = list(itertools.product(args.seeds, args.label_fracs))
    if args.resume:  # each cell is saved as it finishes, so an interrupted run can continue
        cells = [c for c in cells if not _cell_paths(cells_dir, *c)[0].exists()]
        print(f"[resume] {len(cells)} cell(s) left")

    def save(cell, out):
        rows, sel_rows = out
        res_p, sel_p = _cell_paths(cells_dir, *cell)
        if sel_rows:
            pd.DataFrame(sel_rows).to_csv(sel_p, index=False)
        pd.DataFrame(rows).to_csv(res_p, index=False)  # written last: marks the cell done

    t_start = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init,
                                 initargs=(args.threads,), mp_context=mp.get_context("spawn")) as ex:
            futures = {ex.submit(run_cell, args, R, y, s, f, device): (s, f) for s, f in cells}
            for fut in as_completed(futures):
                save(futures[fut], fut.result())
    else:
        _worker_init(args.threads)
        for s, f in cells:
            save((s, f), run_cell(args, R, y, s, f, device))
    print(f"[time] {len(cells)} cell(s) in {time.time() - t_start:.0f}s")

    wanted = set(itertools.product(args.seeds, args.label_fracs))
    results = pd.concat([pd.read_csv(_cell_paths(cells_dir, s, f)[0]) for s, f in sorted(wanted)
                         if _cell_paths(cells_dir, s, f)[0].exists()], ignore_index=True)
    results.to_csv(out_dir / "results.csv", index=False)
    sels = [pd.read_csv(p) for s, f in sorted(wanted) if (p := _cell_paths(cells_dir, s, f)[1]).exists()]
    if sels:
        pd.concat(sels, ignore_index=True).to_csv(out_dir / "selection.csv", index=False)
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
