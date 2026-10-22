"""
Alternating CF–selection: greedy ensemble selection and label-steered factorization.

Doc 08: a smooth in-model combiner trained with cross-entropy selects the wrong base
models, while greedy ensemble selection on R̂ works. This keeps the greedy selector and
still lets labels steer the factorization:

    init:   two-stage factorization (ρ = 1)
    repeat: α ← ensemble_selection(R̂_L)                       (greedy, maximizes PR-AUC)
            train X, b_u, Y on ρ·L_recon + (1−ρ)·L_rank(αᵀR̂_L), α frozen,
            stop-gradient on the labeled instance factors        (no labeled-point distortion)
    final:  α ← ensemble_selection(R̂_L); predict αᵀR̂

L_rank is a pairwise AUC surrogate, so the supervised step targets the same thing the
selector does: ranking positives above negatives.
"""

from typing import Tuple

import numpy as np
import torch

from combiners import ensemble_selection
from cfensemble.data import EnsembleData
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer


def alternating_cf_selection(
    R: np.ndarray, y: np.ndarray, L: np.ndarray, C: np.ndarray, d: int, link: str, rho: float,
    seed: int, rounds: int = 5, epochs_per_round: int = 200, lr: float = 0.01,
    lambda_reg: float = 0.01, pairwise_scale: float = 20.0, init_epochs: int = 3000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, R_hat_final, R_hat_init)."""
    m, n = R.shape
    labels = np.full(n, np.nan)
    labels[L] = y[L]

    tr = CFEnsemblePyTorchTrainer(
        n_classifiers=m, latent_dim=d, rho=1.0, lambda_reg=lambda_reg, aggregator_type="simplex",
        max_epochs=init_epochs, lr=lr, patience=50, min_delta=1e-7, normalize_recon=True,
        link=link, verbose=False, random_seed=seed,
    ).fit(EnsembleData(R, labels, C=C))
    model, device = tr.model, tr.device
    R_hat_init = model.reconstruct().detach().cpu().numpy()

    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    R_t, C_t, labels_t = as_t(R), as_t(C), as_t(labels)
    mask_t = torch.as_tensor(~np.isnan(labels), device=device)
    model.theta.requires_grad_(False)  # α comes from the greedy selector, not gradients
    opt = torch.optim.Adam(model.factor_parameters(), lr=lr)

    R_hat = R_hat_init
    for _ in range(rounds):
        alpha = ensemble_selection(R_hat[:, L], y[L])
        with torch.no_grad():
            model.theta.copy_(torch.log(as_t(alpha) + 1e-6))  # softmax(log α) ≈ α
        for _ in range(epochs_per_round):
            opt.zero_grad()
            loss, _ = model.compute_loss(
                R_t, C_t, labels_t, mask_t, rho=rho, lambda_reg=lambda_reg,
                use_class_weights=True, normalize_recon=True, sup_updates_instances=False,
                sup_loss_type="pairwise", pairwise_scale=pairwise_scale,
            )
            loss.backward()
            opt.step()
        R_hat = model.reconstruct().detach().cpu().numpy()

    return ensemble_selection(R_hat[:, L], y[L]), R_hat, R_hat_init
