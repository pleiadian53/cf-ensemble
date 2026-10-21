"""
Non-CF baseline combiners for the realistic benchmark.

- ``ensemble_selection``: Caruana et al. (2004) greedy forward selection with
  replacement, maximizing PR-AUC on the labeled set. A strong practical baseline.
- ``summa``: label-free spectral combiner for ranked predictions (Ahsen, Vogel &
  Stolovitzky 2019, JMLR; the rank-based form of Parisi et al.'s 2014 Spectral
  Meta-Learner). If classifiers err independently given the class, the covariance
  of their rank-transformed scores is rank-one off the diagonal, with loadings
  proportional to (AUC_u − 0.5); those loadings are the weights.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def ensemble_selection(R_L: np.ndarray, y_L: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Return classifier weights (m,), summing to 1, chosen greedily on labeled columns."""
    m = R_L.shape[0]
    counts = np.zeros(m)
    total = np.zeros(R_L.shape[1])
    for t in range(1, n_iter + 1):
        gains = [average_precision_score(y_L, (total + R_L[u]) / t) for u in range(m)]
        u = int(np.argmax(gains))
        counts[u] += 1
        total += R_L[u]
    return counts / counts.sum()


def summa(R: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """Label-free ensemble scores (n,) from the rank covariance of R (m × n)."""
    m, n = R.shape
    ranks = np.vstack([rankdata(r) / n - 0.5 for r in R])  # higher = more positive
    Q = np.cov(ranks)
    # Rank-one fit to the off-diagonal: iteratively replace the diagonal with v_u².
    Q_fit = Q.copy()
    for _ in range(n_iter):
        vals, vecs = np.linalg.eigh(Q_fit)
        v = vecs[:, -1] * np.sqrt(max(vals[-1], 0.0))
        np.fill_diagonal(Q_fit, v ** 2)
    if v.sum() < 0:  # orient so the majority of classifiers count as better than random
        v = -v
    return v @ ranks
