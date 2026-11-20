"""
Fold-in: an instance's latent vector computed from its column of predictions alone.

With the classifier factors held fixed, the part of the CF-Ensemble objective that involves
one instance i (whose label is not used) is a small convex problem in that instance's
parameters. Fold-in solves it. The same computation serves three purposes in the benchmark:

1. inductive prediction for instances that were not present during the fit;
2. label-blind vectors for labeled instances, so a read-out trained on them sees the same
   kind of input it will see on test instances (episode training);
3. a differentiable map from the classifier factors to instance vectors: every step below is
   a torch operation, so gradients reach X (and b_u) through the fold-in.

Identity link (squared-error reconstruction), per instance i:

    minimize  sum_u c_ui (r_ui - x_u^T y_i)^2 + lam |y_i|^2

    solution  y_i = (X C_i X^T + lam I)^{-1} X C_i r_i,    C_i = diag(c_1i, ..., c_mi)

This is a weighted ridge regression with one data point per classifier (input x_u, target
r_ui, weight c_ui), and it is exactly the instance update of ALS
(``cfensemble.optimization.als.update_instance_factors``).

Logit link (cross-entropy reconstruction of sigmoid(x_u^T y_i + b_u + b_i)), per instance i:

    minimize  sum_u c_ui BCE(r_ui, sigmoid(x_u^T y_i + b_u + b_i)) + lam (|y_i|^2 + b_i^2)

This is an L2-penalized logistic regression with fractional targets r_ui, one data point per
classifier, with the instance bias b_i as an intercept and b_u as a fixed offset. It has no
closed form; it is convex, and Newton's method solves it in a few steps. Gradients pass
through the unrolled Newton steps.

The penalty weight ``lam`` must match how the trainer weights reconstruction against the
penalty for one instance. In ``CFEnsemblePyTorchTrainer`` the penalty sits inside the
rho-weighted term (and ``normalize_recon`` divides both by the same m*n), so rho cancels and
``lam = lambda_reg``. In the ALS trainer it is also ``lambda_reg``.

Shapes: X is (d, m), R and C are (m, n), the result Y is (d, n). d is the latent dimension,
m the number of classifiers, n the number of instances folded in.
"""

from typing import Optional, Tuple

import torch


def fold_in_identity(
    X: torch.Tensor, R: torch.Tensor, C: Optional[torch.Tensor] = None, lam: float = 0.01,
) -> torch.Tensor:
    """Instance vectors for the identity link: Y[:, i] = (X C_i X^T + lam I)^{-1} X C_i r_i.

    With ``C`` None (uniform weights) all instances share one d x d system and a single solve
    handles every column. With per-cell weights every instance has its own d x d system,
    solved as a batch.
    """
    d = X.shape[0]
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    if C is None:
        return torch.linalg.solve(X @ X.T + lam * eye, X @ R)  # (d, n)
    # A[i] = X diag(C[:, i]) X^T + lam I, shape (n, d, d); rhs[i] = X (C[:, i] * R[:, i])
    A = torch.einsum('au,un,bu->nab', X, C, X) + lam * eye
    rhs = (X @ (C * R)).T.unsqueeze(-1)  # (n, d, 1)
    return torch.linalg.solve(A, rhs).squeeze(-1).T  # (d, n)


def fold_in_logit(
    X: torch.Tensor, b_u: torch.Tensor, R: torch.Tensor, C: Optional[torch.Tensor] = None,
    lam: float = 0.01, n_steps: int = 25, tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Instance vectors and instance biases for the logit link, by batched Newton steps.

    Parameters
    ----------
    X : (d, m) classifier factors.
    b_u : (m, 1) or (m,) classifier biases.
    R : (m, n) predicted probabilities, the reconstruction targets.
    C : (m, n) confidence weights, or None for uniform.
    lam : the penalty on y_i and b_i (the trainer's ``lambda_reg``).
    n_steps : maximum Newton steps; ``tol`` stops early once every step is tiny.

    Returns
    -------
    Y : (d, n) and b_i : (1, n), the same shapes as the trainer's parameters.
    """
    d, m = X.shape
    n = R.shape[1]
    if C is None:
        C = torch.ones_like(R)
    # augmented inputs: x~_u = (x_u, 1), so the last coordinate of the unknown is b_i
    Xa = torch.cat([X, torch.ones(1, m, dtype=X.dtype, device=X.device)], dim=0)  # (d+1, m)
    offset = b_u.reshape(m, 1)
    eye = torch.eye(d + 1, dtype=X.dtype, device=X.device)
    theta = torch.zeros(d + 1, n, dtype=X.dtype, device=X.device)
    for _ in range(n_steps):
        P = torch.sigmoid(Xa.T @ theta + offset)  # (m, n)
        grad = Xa @ (C * (P - R)) + 2 * lam * theta  # (d+1, n)
        Wt = C * P * (1 - P)  # (m, n) Newton weights
        H = torch.einsum('au,un,bu->nab', Xa, Wt, Xa) + 2 * lam * eye  # (n, d+1, d+1)
        step = torch.linalg.solve(H, grad.T.unsqueeze(-1)).squeeze(-1).T  # (d+1, n)
        theta = theta - step
        if step.detach().abs().max() < tol:
            break
    return theta[:d], theta[d:]


def fold_in(
    X: torch.Tensor, R: torch.Tensor, C: Optional[torch.Tensor] = None, lam: float = 0.01,
    link: str = 'identity', b_u: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch on the link. Returns (Y, b_i); b_i is None for the identity link."""
    if link == 'identity':
        return fold_in_identity(X, R, C, lam), None
    if link == 'logit':
        if b_u is None:
            raise ValueError("the logit link needs the classifier biases b_u")
        return fold_in_logit(X, b_u, R, C, lam)
    raise ValueError(f"link must be 'identity' or 'logit', got {link!r}")


def reconstruct(
    X: torch.Tensor, Y: torch.Tensor, link: str = 'identity',
    b_u: Optional[torch.Tensor] = None, b_i: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """R_hat from factors, matching ``CFEnsembleNet.reconstruct`` for either link."""
    Z = X.T @ Y
    if link == 'logit':
        return torch.sigmoid(Z + b_u.reshape(-1, 1) + b_i.reshape(1, -1))
    return Z
