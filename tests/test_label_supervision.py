"""Tests for examples/real_world/label_supervision.py: no test label is ever read, inductive
predictions are independent of the other test instances, episodes hide both classes."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "real_world"))
from label_supervision import (  # noqa: E402
    FitConfig, _stratified_hide, label_row, matched_joint, two_stage_R_hat,
)

CFG = FitConfig(d=3, max_epochs=150, refine_epochs=80, patience=1000, refit_g_steps=20, rho=0.5)


def toy(seed=0, m=6, n=240):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    skill = rng.uniform(0.05, 0.35, size=(m, 1))
    R = np.clip(0.5 + skill * (2 * y - 1) + 0.15 * rng.standard_normal((m, n)), 0.01, 0.99)
    C = 0.3 + rng.random((m, n))
    L, T = np.arange(0, 120), np.arange(120, n)
    return R, C, y, L, T


ARMS = [
    ("label_row", dict(episodes=False, inductive=False)),
    ("label_row_sg", dict(episodes=False, inductive=False, blind=True)),
    ("episodes", dict(episodes=True, inductive=False)),
    ("label_row_ind", dict(episodes=False, inductive=True)),
    ("episodes_ind", dict(episodes=True, inductive=True)),
]


@pytest.mark.parametrize("name,kw", ARMS)
def test_test_labels_never_read(name, kw):
    R, C, y, L, T = toy()
    y2 = y.copy()
    y2[T] = 1 - y2[T]  # flip every hidden test label
    a = label_row(R, C, y, L, T, CFG, seed=0, **kw)[0]
    b = label_row(R, C, y2, L, T, CFG, seed=0, **kw)[0]
    np.testing.assert_array_equal(a, b)


def test_matched_joint_test_labels_never_read():
    R, C, y, L, T = toy()
    y2 = y.copy()
    y2[T] = 1 - y2[T]
    np.testing.assert_array_equal(matched_joint(R, C, y, L, T, CFG, 0)[0],
                                  matched_joint(R, C, y2, L, T, CFG, 0)[0])


@pytest.mark.parametrize("episodes", [False, True])
def test_inductive_is_independent_of_other_test_instances(episodes):
    """Changing some test instances' predictions leaves the others' scores unchanged."""
    R, C, y, L, T = toy()
    R2 = R.copy()
    R2[:, T[:60]] = np.random.default_rng(1).random((R.shape[0], 60))
    a = label_row(R, C, y, L, T, CFG, seed=0, episodes=episodes, inductive=True)[0]
    b = label_row(R2, C, y, L, T, CFG, seed=0, episodes=episodes, inductive=True)[0]
    np.testing.assert_allclose(a[60:], b[60:], atol=1e-6)
    assert np.abs(a[:60] - b[:60]).max() > 1e-3  # the changed ones do change


def test_transductive_depends_on_other_test_instances():
    R, C, y, L, T = toy()
    R2 = R.copy()
    R2[:, T[:60]] = np.random.default_rng(1).random((R.shape[0], 60))
    a = label_row(R, C, y, L, T, CFG, seed=0, episodes=False, inductive=False)[0]
    b = label_row(R2, C, y, L, T, CFG, seed=0, episodes=False, inductive=False)[0]
    assert np.abs(a[60:] - b[60:]).max() > 1e-6


def test_two_stage_inductive_independent():
    R, C, y, L, T = toy()
    R2 = R.copy()
    R2[:, T[:60]] = 0.5
    a = two_stage_R_hat(R, C, L, T, CFG, 0, inductive=True)
    b = two_stage_R_hat(R2, C, L, T, CFG, 0, inductive=True)
    np.testing.assert_allclose(a[:, T[60:]], b[:, T[60:]], atol=1e-6)
    np.testing.assert_allclose(a[:, L], b[:, L], atol=1e-6)


def test_stratified_hide():
    rng = np.random.default_rng(0)
    y_L = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(20):
        h = _stratified_hide(y_L, 0.2, rng)
        assert h[y_L == 1].sum() == 1 and (~h[y_L == 1]).sum() == 1
        assert h[y_L == 0].sum() == 2


@pytest.mark.parametrize("name,kw", ARMS)
def test_learns_something(name, kw):
    R, C, y, L, T = toy()
    scores = label_row(R, C, y, L, T, CFG, seed=0, **kw)[0]
    assert roc_auc_score(y[T], scores) > 0.8


def test_inductive_column_order():
    """Unsorted L gives the same R_hat and scores as sorted L (columns written back in place)."""
    R, C, y, L, T = toy()
    L_perm = np.random.default_rng(3).permutation(L)
    np.testing.assert_allclose(two_stage_R_hat(R, C, L, T, CFG, 0, inductive=True),
                               two_stage_R_hat(R, C, L_perm, T, CFG, 0, inductive=True), atol=1e-6)
    a = label_row(R, C, y, L, T, CFG, seed=0, episodes=False, inductive=True)
    b = label_row(R, C, y, L_perm, T, CFG, seed=0, episodes=False, inductive=True)
    np.testing.assert_allclose(a[0], b[0], atol=1e-6)
    np.testing.assert_allclose(a[1], b[1], atol=1e-6)


def test_blind_label_row_cannot_move_instance_vectors():
    """With every label term label-blind and no episodes, the labels never reach X, Y or the
    biases: the factors after refinement equal those of refining with the label term off."""
    import torch
    from label_supervision import LabelRowCF, factors_of, two_stage
    R, C, y, L, T = toy()
    cfg = FitConfig(**{**CFG.__dict__, "rho": 0.5})
    warm = factors_of(two_stage(R, C, np.arange(R.shape[1]), cfg, 0))
    _, Rh_a, _, _ = label_row(R, C, y, L, T, cfg, 0, episodes=False, inductive=False, warm=warm,
                              blind=True)
    y2 = y.copy()
    y2[L] = 1 - y2[L]  # different labels: the classifier-row factors must not care
    _, Rh_b, _, _ = label_row(R, C, y2, L, T, cfg, 0, episodes=False, inductive=False, warm=warm,
                              blind=True)
    np.testing.assert_allclose(Rh_a, Rh_b, atol=1e-6)
