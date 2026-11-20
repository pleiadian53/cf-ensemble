"""Tests for examples/real_world/selection.py: folds, choosing the better candidate, and that
selection reads no test label and no held-out label while fitting."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "real_world"))
from label_supervision import FitConfig  # noqa: E402
from selection import (  # noqa: E402
    confirmed_scorer, cv_select, grid, joint_scorer, label_row_scorer, stratified_folds,
)

CFG = FitConfig(d=3, max_epochs=120, refine_epochs=60, patience=1000, refit_g_steps=20)


def toy(seed=0, m=6, n=240):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    skill = rng.uniform(0.05, 0.35, size=(m, 1))
    R = np.clip(0.5 + skill * (2 * y - 1) + 0.15 * rng.standard_normal((m, n)), 0.01, 0.99)
    L, T = np.arange(0, 120), np.arange(120, n)
    return R, y, L, T


def test_folds_cover_and_stratify():
    y_L = np.array([1] * 5 + [0] * 40)
    folds = stratified_folds(y_L, 3, seed=0)
    assert len(folds) == 3
    val = np.concatenate([va for _, va in folds])
    assert sorted(val) == list(range(len(y_L)))
    for tr, va in folds:
        assert not set(tr) & set(va)
        assert y_L[va].sum() >= 1 and (1 - y_L[va]).sum() >= 1


def test_folds_shrink_with_few_positives():
    y_L = np.array([1, 1] + [0] * 30)
    assert len(stratified_folds(y_L, 5, seed=0)) == 2


def test_picks_the_informative_candidate():
    R, y, L, _ = toy()
    rng = np.random.default_rng(0)

    def fit_score(cand, L_tr, L_val, ctx):
        return R[:, L_val].mean(axis=0) if cand == "signal" else rng.random(len(L_val))

    best, table = cv_select(["noise", "signal"], fit_score, y, L)
    assert best == "signal" and len(table) == 2


def test_ties_go_to_the_first_candidate():
    R, y, L, _ = toy()
    best, _ = cv_select(["a", "b"], lambda c, tr, va, ctx: R[0, va], y, L)
    assert best == "a"


def _scorers(R, y):
    return {
        "confirmed": (confirmed_scorer(R, y, CFG, 0), grid(["uniform", "classifier"], [2, 3])),
        "confirmed_ind": (confirmed_scorer(R, y, CFG, 0, inductive=True), grid(["classifier"], [3])),
        "joint": (joint_scorer(R, y, "classifier", CFG, 0), [0.5, 0.9]),
        "episodes": (label_row_scorer(R, y, "classifier", CFG, 0, episodes=True, inductive=False,
                                      blind=True), grid([0.5], [0.2, 0.3])),
        "label_row_sg_ind": (label_row_scorer(R, y, "classifier", CFG, 0, episodes=False,
                                              inductive=True, blind=True), [0.5]),
    }


@pytest.mark.parametrize("name", ["confirmed", "confirmed_ind", "joint", "episodes", "label_row_sg_ind"])
def test_selection_ignores_test_labels(name):
    R, y, L, T = toy()
    y2 = y.copy()
    y2[T] = 1 - y2[T]
    f1, cands = _scorers(R, y)[name]
    f2, _ = _scorers(R, y2)[name]
    b1, t1 = cv_select(cands, f1, y, L)
    b2, t2 = cv_select(cands, f2, y2, L)
    assert b1 == b2
    np.testing.assert_allclose(t1.oof_pr_auc, t2.oof_pr_auc)


@pytest.mark.parametrize("name", ["confirmed", "joint", "episodes"])
def test_held_out_labels_do_not_shape_their_scores(name):
    """Flipping the labels of the validation fold leaves that fold's scores unchanged."""
    R, y, L, _ = toy()
    tr, va = stratified_folds(y[L], 3, seed=0)[0]
    L_tr, L_val = L[tr], L[va]
    y2 = y.copy()
    y2[L_val] = 1 - y2[L_val]
    f1, cands = _scorers(R, y)[name]
    f2, _ = _scorers(R, y2)[name]
    np.testing.assert_allclose(f1(cands[0], L_tr, L_val, {}), f2(cands[0], L_tr, L_val, {}), atol=1e-6)


def test_uniform_transductive_factorization_is_fitted_once(monkeypatch):
    """Uniform confidence reads no label, so one transductive fit serves every fold."""
    import selection
    calls = []
    real = selection.two_stage
    monkeypatch.setattr(selection, "two_stage", lambda *a, **k: calls.append(1) or real(*a, **k))
    R, y, L, _ = toy()
    cv_select(grid(["uniform"], [3]), confirmed_scorer(R, y, CFG, 0), y, L)
    assert len(calls) == 1
    calls.clear()
    cv_select(grid(["classifier"], [3]), confirmed_scorer(R, y, CFG, 0), y, L)
    assert len(calls) == 3  # label-informed confidence: one fit per fold
