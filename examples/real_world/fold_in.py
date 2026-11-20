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
closed form; it is convex, and Newton's method with a backtracking line search solves it
in a few steps. Gradients come from one differentiable Newton step at the solution.

The penalty weight ``lam`` must match how the trainer weights reconstruction against the
penalty for one instance. In ``CFEnsemblePyTorchTrainer`` the penalty sits inside the
rho-weighted term (and ``normalize_recon`` divides both by the same m*n), so rho cancels and
``lam = lambda_reg``. In the ALS trainer it is also ``lambda_reg``.

Shapes: X is (d, m), R and C are (m, n), the result Y is (d, n). d is the latent dimension,
m the number of classifiers, n the number of instances folded in.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


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
    lam: float = 0.01, n_steps: int = 50, tol: float = 1e-8, max_halvings: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Instance vectors and instance biases for the logit link, by damped Newton steps.

    Each instance's problem is convex, but a full Newton step from zero can overshoot badly
    when the factors or the classifier biases are large (seen after label-supervised
    refinement: parameters of order 1e3 and no convergence). Each step is therefore damped
    by a per-instance backtracking line search (Armijo condition): the step length is
    halved, for each instance separately, until its objective decreases enough. The step
    iterations run without gradient tracking; one final full Newton step is then taken with
    gradients, which at the optimum gives the exact (implicit-function) derivative of the
    solution with respect to X and b_u, without storing the iterations.

    Parameters
    ----------
    X : (d, m) classifier factors.
    b_u : (m, 1) or (m,) classifier biases.
    R : (m, n) predicted probabilities, the reconstruction targets.
    C : (m, n) confidence weights, or None for uniform.
    lam : the penalty on y_i and b_i (the trainer's ``lambda_reg``).
    n_steps : maximum Newton steps; ``tol`` stops early once every step is tiny.
    max_halvings : maximum step halvings per line search.

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

    def objective(theta):  # per-instance objective, shape (n,)
        Z = Xa.T @ theta + offset
        bce = F.binary_cross_entropy_with_logits(Z, R, reduction='none')
        return (C * bce).sum(dim=0) + lam * (theta ** 2).sum(dim=0)

    def newton_step(theta):
        P = torch.sigmoid(Xa.T @ theta + offset)  # (m, n)
        grad = Xa @ (C * (P - R)) + 2 * lam * theta  # (d+1, n)
        Wt = C * P * (1 - P)  # (m, n) Newton weights
        H = torch.einsum('au,un,bu->nab', Xa, Wt, Xa) + 2 * lam * eye  # (n, d+1, d+1)
        return torch.linalg.solve(H, grad.T.unsqueeze(-1)).squeeze(-1).T, grad  # (d+1, n)

    # 1. Solve without gradient tracking: damped Newton with a per-instance line search.
    #    Near the optimum the decrease falls below floating-point resolution, so the
    #    Armijo test allows a few ulps of the objective.
    slack = 8 * torch.finfo(X.dtype).eps
    with torch.no_grad():
        theta = torch.zeros(d + 1, n, dtype=X.dtype, device=X.device)
        for _ in range(n_steps):
            step, grad = newton_step(theta)
            f0 = objective(theta)
            slope = (grad * step).sum(dim=0)  # directional decrease, >= 0 (H is pos. def.)
            t = torch.ones(n, dtype=X.dtype, device=X.device)
            for _ in range(max_halvings):
                ok = objective(theta - t * step) <= f0 - 1e-4 * t * slope + slack * f0.abs()
                if bool(ok.all()):
                    break
                t = torch.where(ok, t, t / 2)
            theta = theta - t * step
            if (t * step).abs().max() < tol:
                break

    # 2. One full Newton step from the solution, with gradients. At the optimum the step is
    #    ~0, and its derivative with respect to X and b_u is the implicit-function-theorem
    #    derivative of the minimizer: d theta*/dX = -H^{-1} d(grad)/dX.
    if X.requires_grad or b_u.requires_grad or R.requires_grad or C.requires_grad:
        theta = theta - newton_step(theta)[0]
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
