"""Tests for examples/real_world/fold_in.py: correctness, agreement with ALS and with the
PyTorch trainer's own loss, and gradients through the fold-in."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "real_world"))
from fold_in import fold_in, fold_in_identity, fold_in_logit, reconstruct  # noqa: E402

from cfensemble.optimization.als import update_instance_factors
from cfensemble.optimization.trainer_pytorch import CFEnsembleNet

D, M, N, LAM = 3, 6, 9, 0.05


@pytest.fixture
def problem():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(D, M, generator=g, dtype=torch.float64)
    b_u = 0.3 * torch.randn(M, 1, generator=g, dtype=torch.float64)
    R = torch.rand(M, N, generator=g, dtype=torch.float64)
    C = 0.2 + torch.rand(M, N, generator=g, dtype=torch.float64)
    return X, b_u, R, C


def test_identity_matches_als_update(problem):
    X, _, R, C = problem
    for weights in (None, C):
        Y = fold_in_identity(X, R, weights, LAM)
        Cn = np.ones((M, N)) if weights is None else weights.numpy()
        Y_als = update_instance_factors(X.numpy(), R.numpy(), Cn, LAM)
        np.testing.assert_allclose(Y.numpy(), Y_als, atol=1e-10)


def test_logit_is_the_minimizer(problem):
    X, b_u, R, C = problem
    Y, b_i = fold_in_logit(X, b_u, R, C, LAM)
    theta = torch.cat([Y, b_i]).clone().requires_grad_()
    Z = X.T @ theta[:D] + b_u + theta[D:]
    obj = (C * torch.nn.functional.binary_cross_entropy_with_logits(Z, R, reduction='none')).sum() \
        + LAM * (theta ** 2).sum()
    obj.backward()
    assert theta.grad.abs().max() < 1e-8


@pytest.mark.parametrize("link", ["identity", "logit"])
def test_stationary_for_the_trainer_loss(problem, link):
    """At the fold-in, the trainer's own loss (rho = 1, no labels) has zero gradient in that
    instance's parameters: the penalty convention (lam = lambda_reg) matches the trainer."""
    X, b_u, R, C = problem
    net = CFEnsembleNet(M, N, D, link=link).double()
    Y, b_i = fold_in(X, R, C, LAM, link=link, b_u=b_u)
    with torch.no_grad():
        net.X.copy_(X)
        net.Y.copy_(Y)
        if link == 'logit':
            net.b_u.copy_(b_u)
            net.b_i.copy_(b_i)
    labels = torch.full((N,), float('nan'), dtype=torch.float64)
    mask = torch.zeros(N, dtype=torch.bool)
    for normalize in (False, True):
        net.zero_grad()
        loss, _ = net.compute_loss(R, C, labels, mask, rho=1.0, lambda_reg=LAM,
                                   normalize_recon=normalize)
        loss.backward()
        tol = 1e-7 / (M * N) if normalize else 1e-7  # normalize_recon divides the loss by m*n
        assert net.Y.grad.abs().max() < tol
        if link == 'logit':
            assert net.b_i.grad.abs().max() < tol


def test_reconstruct_matches_net(problem):
    X, b_u, R, C = problem
    net = CFEnsembleNet(M, N, D, link='logit').double()
    Y, b_i = fold_in(X, R, C, LAM, link='logit', b_u=b_u)
    with torch.no_grad():
        net.X.copy_(X), net.Y.copy_(Y), net.b_u.copy_(b_u), net.b_i.copy_(b_i)
        torch.testing.assert_close(net.reconstruct(), reconstruct(X, Y, 'logit', b_u, b_i))


@pytest.mark.parametrize("link", ["identity", "logit"])
def test_gradient_through_fold_in(problem, link):
    """Gradients reach X through the fold-in (finite-difference check)."""
    _, b_u, R, C = problem
    X0 = problem[0].clone().requires_grad_()

    def f(X):
        Y, b_i = fold_in(X, R, C, LAM, link=link, b_u=b_u)
        return reconstruct(X, Y, link, b_u, b_i).sum(dim=0)

    assert torch.autograd.gradcheck(f, (X0,), eps=1e-6, atol=1e-5)


def _toy_data(seed=0, m=5, n=60):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(float)
    R = np.clip(0.5 + 0.3 * (2 * y - 1) * rng.random((m, 1)) + 0.15 * rng.standard_normal((m, n)), 0, 1)
    labels = y.copy()
    labels[40:] = np.nan
    C = 0.2 + rng.random((m, n))
    return R, labels, C


def test_pytorch_predict_uses_c_new():
    from cfensemble.data import EnsembleData
    from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer
    R, labels, C = _toy_data()
    tr = CFEnsemblePyTorchTrainer(n_classifiers=5, latent_dim=3, max_epochs=20, verbose=False,
                                  random_seed=0, device='cpu')
    tr.fit(EnsembleData(R, labels, C))
    R_new, C_new = R[:, 40:], C[:, 40:]
    X = tr.model.X.detach()
    Y = fold_in_identity(X, torch.from_numpy(R_new).to(X.dtype), torch.from_numpy(C_new).to(X.dtype),
                         tr.lambda_reg)
    with torch.no_grad():
        expected = tr.model.aggregate(X.T @ Y).numpy()
    np.testing.assert_allclose(tr.predict(R_new, C_new), expected, atol=1e-5)
    # without C_new the uniform fold-in is used, which differs when C is not uniform
    assert np.abs(tr.predict(R_new) - expected).max() > 1e-4


def test_als_predict_uses_c_new():
    from cfensemble.data import EnsembleData
    from cfensemble.optimization.trainer import CFEnsembleTrainer
    R, labels, C = _toy_data()
    tr = CFEnsembleTrainer(n_classifiers=5, latent_dim=3, max_iter=5, verbose=False)
    tr.fit(EnsembleData(R, labels, C))
    R_new, C_new = R[:, 40:], C[:, 40:]
    Y = update_instance_factors(tr.X, R_new, C_new, tr.lambda_reg)
    expected = tr.aggregator.predict(tr.X.T @ Y)
    np.testing.assert_allclose(tr.predict(R_new=R_new, C_new=C_new), expected, atol=1e-10)


def test_pytorch_focal_gamma_is_used():
    """focal_gamma changes the supervised loss (it used to be stored and ignored)."""
    net = CFEnsembleNet(4, 10, 2)
    R_hat = torch.rand(4, 10)
    y = (torch.rand(10) < 0.5).float()
    plain = net.supervised_loss(R_hat, y)
    focal = net.supervised_loss(R_hat, y, focal_gamma=2.0)
    assert focal < plain  # every focal factor is below 1
    torch.testing.assert_close(net.supervised_loss(R_hat, y, focal_gamma=0.0), plain)


def test_logit_converges_with_large_factors(problem):
    """Large factors and classifier biases (as after label-supervised refinement) used to make
    undamped Newton diverge; the damped version still reaches the minimum."""
    X, b_u, R, C = problem
    X, b_u = 4.0 * X, 8.0 * b_u + 3.0
    Y, b_i = fold_in_logit(X, b_u, R, C, LAM)
    theta = torch.cat([Y, b_i]).clone().requires_grad_()
    Z = X.T @ theta[:D] + b_u + theta[D:]
    obj = (C * torch.nn.functional.binary_cross_entropy_with_logits(Z, R, reduction='none')).sum() \
        + LAM * (theta ** 2).sum()
    obj.backward()
    assert theta.grad.abs().max() < 1e-6
