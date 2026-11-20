"""
Choosing settings on the labeled set by cross-validation, never on the test set.

Round 4 confirmed the CF method with the confidence type and latent dimension picked per
dataset by hand, after looking at round 3's test scores. A usable method must make those
choices from the labeled set L alone, and so must every arm that has settings of its own
(the reconstruction weight rho, the fraction of labels hidden per episode). This module does
it by stratified K-fold cross-validation inside L:

    for each fold:  L_tr = L minus the fold, L_val = the fold
        for each candidate setting:
            refit the arm as if L_tr were the labeled set (confidence weights included),
            score the instances of L_val
    per candidate:  PR-AUC of the pooled out-of-fold scores over L
    choose:         the best candidate; ties go to the earlier one in the grid

Pooling the out-of-fold scores before computing PR-AUC, instead of averaging per-fold
PR-AUCs, matters at small budgets: at COFFIN's 2% budget L holds about 10 positives, so a
single fold holds about 3, and a per-fold PR-AUC rests on those few.

Every refit sees what the arm sees in deployment, with L_tr in the role of L: a transductive
arm sees the predictions of every instance (L_val and the test set included) and the labels
of L_tr only; an inductive arm is fitted on the columns of L_tr only and folds in L_val.
Nothing reads y outside L_tr while fitting, and the labels of L_val are read only to score.

The selection order used by the benchmark is sequential: the confidence type and latent
dimension are chosen once, with the confirmed method (two-stage factorization, ensemble
selection on R_hat), and shared by every arm; then each label-supervised arm chooses its own
rho (and hidden fraction) with those fixed. Choosing everything jointly per arm would
multiply the cost by the size of the confidence grid.
"""

from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from combiners import ensemble_selection
from confidence import make_confidence
from label_supervision import (
    FitConfig, R_hat_from_net, factors_of, label_row, matched_joint, two_stage,
)

# fit_score(candidate, L_tr, L_val, ctx) -> scores for L_val (higher = more positive).
# ctx is a dict private to one fold, for sharing expensive pieces across candidates.
FitScore = Callable[[object, np.ndarray, np.ndarray, dict], np.ndarray]


def stratified_folds(y_L: np.ndarray, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified folds over positions 0..len(y_L)-1.

    The number of folds is lowered to the size of the smaller class if needed (at least 2),
    so every validation fold holds at least one positive and one negative.
    """
    n_pos = int(y_L.sum())
    k = max(2, min(n_folds, n_pos, len(y_L) - n_pos))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_L)), y_L))


def cv_select(candidates: Sequence, fit_score: FitScore, y: np.ndarray, L: np.ndarray,
              n_folds: int = 3, seed: int = 0) -> Tuple[object, pd.DataFrame]:
    """Return (best candidate, table of pooled out-of-fold PR-AUC per candidate)."""
    folds = stratified_folds(y[L], n_folds, seed)
    oof = {i: np.full(len(L), np.nan) for i in range(len(candidates))}
    for tr, va in folds:
        ctx: dict = {}
        for i, cand in enumerate(candidates):
            oof[i][va] = fit_score(cand, L[tr], L[va], ctx)
    rows = [{"candidate": repr(c), "oof_pr_auc": average_precision_score(y[L], oof[i]),
             "n_folds": len(folds)} for i, c in enumerate(candidates)]
    table = pd.DataFrame(rows)
    best = int(np.argmax(table.oof_pr_auc.to_numpy()))  # argmax keeps the first of ties
    return candidates[best], table


# --- scorers for the benchmark's arms -------------------------------------------------------------

def _confidence(ctx: dict, conf: str, R, y, L_tr, seed) -> np.ndarray:
    """C for this fold, built from L_tr only; cached per fold and kind."""
    cache = ctx.setdefault("conf_cache", {})  # make_confidence's own cache, per labeled set
    key = ("C", conf)
    if key not in ctx:
        ctx[key] = make_confidence(conf, R, y, L_tr, seed, cache)
    return ctx[key]


def _warm(ctx: dict, R, C, L_tr, cfg: FitConfig, seed: int, inductive: bool, conf: str,
          shared: dict):
    """The two-stage net for this fold and configuration, shared across candidates.

    With uniform confidence the transductive factorization reads no label at all, so it is
    the same in every fold: it is fitted once and kept in ``shared`` (one per scorer).
    """
    key = ("net", conf, cfg.d, cfg.link, inductive)
    store = shared if (conf == "uniform" and not inductive) else ctx
    if key not in store:
        cols = np.sort(L_tr) if inductive else np.arange(R.shape[1])
        store[key] = two_stage(R, C, cols, cfg, seed)
    return store[key]


def confirmed_scorer(R, y, base: FitConfig, seed: int, inductive: bool = False) -> FitScore:
    """Candidates (confidence, d): two-stage factorization, ensemble selection on R_hat."""
    shared: dict = {}

    def fit_score(cand, L_tr, L_val, ctx):
        conf, d = cand
        cfg = replace(base, d=d)
        C = _confidence(ctx, conf, R, y, L_tr, seed)
        net = _warm(ctx, R, C, L_tr, cfg, seed, inductive, conf, shared)
        R_hat = R_hat_from_net(net, R, C, L_tr, L_val, cfg, inductive)
        w = ensemble_selection(R_hat[:, L_tr], y[L_tr])
        return w @ R_hat[:, L_val]
    return fit_score


def joint_scorer(R, y, conf: str, base: FitConfig, seed: int) -> FitScore:
    """Candidates rho: the matched joint objective, scored by its own read-out g."""
    def fit_score(rho, L_tr, L_val, ctx):
        C = _confidence(ctx, conf, R, y, L_tr, seed)
        return matched_joint(R, C, y, L_tr, L_val, replace(base, rho=rho), seed)[0]
    return fit_score


def label_row_scorer(R, y, conf: str, base: FitConfig, seed: int, episodes: bool,
                     inductive: bool, blind: bool) -> FitScore:
    """Candidates rho, or (rho, hide_frac) for episodes: the label-row read-out."""
    shared: dict = {}

    def fit_score(cand, L_tr, L_val, ctx):
        rho, hide = cand if episodes else (cand, base.hide_frac)
        cfg = replace(base, rho=rho, hide_frac=hide)
        C = _confidence(ctx, conf, R, y, L_tr, seed)
        warm = factors_of(_warm(ctx, R, C, L_tr, cfg, seed, inductive, conf, shared))
        return label_row(R, C, y, L_tr, L_val, cfg, seed, episodes=episodes,
                         inductive=inductive, warm=warm, blind=blind)[0]
    return fit_score


def grid(*axes: Sequence) -> List[Tuple]:
    """All combinations of the given axes, in order (the first is the tie-break winner)."""
    out: List[Tuple] = [()]
    for axis in axes:
        out = [t + (v,) for t in out for v in axis]
    return out


def describe(table: pd.DataFrame, best) -> Dict[str, object]:
    """A compact record of one selection, for the results file."""
    return {"selected": repr(best), "oof_pr_auc": float(table.oof_pr_auc.max()),
            "n_candidates": len(table), "n_folds": int(table.n_folds.iloc[0])}
