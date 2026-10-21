"""
Confidence matrices C for the realistic benchmark: which entries of R to trust.

In the reconstruction loss Σ c_ui (r_ui − r̂_ui)², a low c_ui turns entry (u, i) into
something close to a missing value: instance factors are shaped by the trusted
entries, and the low-confidence entry is filled in from the low-rank structure.
A classifier with low confidence everywhere stops shaping the latent space at all.

- ``uniform``:    all ones (baseline)
- ``classifier``: c_ui = q_u, per-classifier quality from labeled cells,
                  q_u ∝ max(AUC_u − 0.5, 0): the soft analogue of ensemble
                  selection dropping a useless model
- ``learned``:    cell-level reliability (``ReliabilityWeightModel``), class-balanced
                  so positives count (under heavy imbalance a classifier predicting ~0
                  everywhere otherwise looks reliable), and cross-fitted on L so the
                  labeled cells' confidences are out-of-sample, like the test cells'
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cfensemble.models.reliability import ReliabilityWeightModel

MIN_CONFIDENCE = 0.05  # keep every entry minimally in play (and C strictly positive)


def classifier_auc(R: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y[idx], R[u, idx]) for u in range(R.shape[0])])


def classifier_confidence(R: np.ndarray, y: np.ndarray, L: np.ndarray) -> np.ndarray:
    """c_ui = q_u with q_u = max(AUC_u(L) − 0.5, 0), scaled so the best classifier gets 1."""
    q = np.clip(classifier_auc(R, y, L) - 0.5, 0.0, None)
    q = q / q.max() if q.max() > 0 else np.ones_like(q)
    return np.repeat(np.maximum(q, MIN_CONFIDENCE)[:, None], R.shape[1], axis=1)


def learned_confidence(
    R: np.ndarray, y: np.ndarray, L: np.ndarray, seed: int,
    n_folds: int = 3, neg_per_pos: int = 20,
) -> np.ndarray:
    """Cell-level reliability, cross-fitted on L and fitted on all of L for the rest.

    Negative training instances are subsampled to ``neg_per_pos`` per positive to keep
    the GBM cheap; class-balanced weights keep the two classes equal either way.
    """
    m, n = R.shape
    rng = np.random.default_rng(seed)
    C = np.empty_like(R)

    def fit_on(train: np.ndarray) -> Tuple[ReliabilityWeightModel, dict]:
        pos, neg = train[y[train] == 1], train[y[train] == 0]
        cap = neg_per_pos * max(len(pos), 1)
        if len(neg) > cap:
            neg = rng.choice(neg, cap, replace=False)
        keep = np.concatenate([pos, neg])
        mask = np.zeros(n, dtype=bool)
        mask[keep] = True
        labels = np.full(n, np.nan)
        labels[keep] = y[keep]
        stats = {"auc_L": classifier_auc(R, y, train)}
        model = ReliabilityWeightModel(model_type="gbm", random_state=seed)
        model.fit(R, labels, mask, stats, class_balanced=True)
        return model, stats

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in folds.split(L, y[L]):
        model, stats = fit_on(L[tr])
        C[:, L[te]] = model.predict(R[:, L[te]], stats)

    rest = np.setdiff1d(np.arange(n), L)
    model, stats = fit_on(L)
    C[:, rest] = model.predict(R[:, rest], stats)
    return np.maximum(C, MIN_CONFIDENCE)


def make_confidence(kind: str, R: np.ndarray, y: np.ndarray, L: np.ndarray, seed: int) -> np.ndarray:
    if kind == "uniform":
        return np.ones_like(R)
    if kind == "classifier":
        return classifier_confidence(R, y, L)
    if kind == "learned":
        return learned_confidence(R, y, L, seed)
    raise ValueError(f"unknown confidence {kind!r}")
