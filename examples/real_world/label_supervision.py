"""
Ways to let the known labels shape the CF factorization without a train/test mismatch.

The rule every arm here obeys or is measured against: a vector that is graded against a
label must be label-blind, computed the way a test instance's vector is. Otherwise a
read-out is trained on one kind of vector and applied to another. The fold-in that makes
vectors label-blind, and its derivation, are in fold_in.py.

All arms use the confirmed ingredients (doc 07): the logit link with classifier and
instance biases, a label-informed confidence matrix C, ``normalize_recon``. Arms:

- ``two_stage``: the confirmed factorization (rho = 1, labels only through C).
  Transductive: fitted on every column. Inductive: fitted on the labeled columns only,
  test instances folded in from their predictions with their own confidence weights.
- ``matched_joint`` (option 1): the PyTorch trainer with rho < 1 and stop-gradient on the
  labeled instance factors, so labels move X, b_u and g, never an instance vector; g gets
  its own learning rate and a convex refit (it was under-trained in doc 04).
- ``label_row`` (option 2): the labels as one more row of R, with its own vector x_label
  and bias; a test label is the missing entry sigmoid(x_label . y_i + b_label + b_i).
  No separate aggregator. Labeled instances' vectors see their label-row entry, so this
  arm breaks the rule; it measures what the mismatch costs.
- ``label_row_sg``: the label row with stop-gradient on every instance vector (and
  instance bias): all vectors are label-blind. The label row cannot reach X at all here
  (its logit does not involve X), so this is a logistic read-out on the two-stage latent
  vectors, the reference for what the fold-in path adds.
- ``episodes`` (option 3): the label-row model trained the way it is used. At every step a
  stratified random subset H of the labeled instances is hidden: their vectors are folded
  in from the classifier rows alone (label-blind, differentiable in X and b_u) and their
  label loss goes through those vectors, the one path by which labels reach X. The kept
  instances' label terms use their free vectors with stop-gradient, so every vector the
  read-out is trained on is label-blind. Transductive or inductive.

  (A first version let the kept instances' vectors see their labels, as option 2 does. The
  protein_homo smoke run showed the cost: the read-out trained on memorized vectors, PR-AUC
  1.0 on L, and in the inductive form X bent until test fold-ins reconstructed six times
  worse than at the warm start.)

Every arm also exposes its reconstructed classifier rows R_hat, so ensemble selection on
R_hat can be scored as a second read-out, separating "is R_hat better?" from "is the
in-model read-out better?".

Indices: ``L`` labeled and ``T`` test column indices into R (m x n). Only y[L] is ever read.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fold_in import fold_in, reconstruct
from cfensemble.data import EnsembleData
from cfensemble.optimization.trainer_pytorch import CFEnsemblePyTorchTrainer


@dataclass
class FitConfig:
    d: int = 8
    link: str = "logit"
    lambda_reg: float = 0.01
    lr: float = 0.01
    max_epochs: int = 3000
    patience: int = 50
    min_delta: float = 1e-7
    rho: float = 0.5           # weight of reconstruction vs label loss (label-supervised arms)
    hide_frac: float = 0.2     # episodes: fraction of each class hidden per step
    refine_epochs: int = 1000  # label-row / episode training after the two-stage warm start
    lr_g: float = 0.05         # matched_joint: g's own learning rate
    refit_g_steps: int = 500   # matched_joint: convex refit of g at the end
    device: str = "cpu"


def _t(a, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


def _balanced_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Class-balanced mean BCE, the same weighting as the trainers' supervised loss."""
    n, n_pos = len(y), y.sum()
    if 0 < n_pos < n:
        w = torch.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))
    else:
        w = torch.ones_like(y)
    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction="none")).sum() / w.sum()


# --- two-stage: the confirmed factorization, transductive or inductive --------------------------

def two_stage(R: np.ndarray, C: np.ndarray, cols: np.ndarray, cfg: FitConfig, seed: int):
    """Fit the confirmed factorization (rho = 1) on the columns ``cols``; return the net."""
    tr = CFEnsemblePyTorchTrainer(
        n_classifiers=R.shape[0], latent_dim=cfg.d, rho=1.0, lambda_reg=cfg.lambda_reg,
        max_epochs=cfg.max_epochs, lr=cfg.lr, patience=cfg.patience, min_delta=cfg.min_delta,
        normalize_recon=True, link=cfg.link, verbose=False, random_seed=seed, device=cfg.device,
    )
    tr.fit(EnsembleData(R[:, cols], np.full(len(cols), np.nan), C=C[:, cols]))
    return tr.model


def factors_of(net) -> Dict[str, torch.Tensor]:
    out = {"X": net.X.detach().clone(), "Y": net.Y.detach().clone()}
    if net.link == "logit":
        out["b_u"] = net.b_u.detach().clone()
        out["b_i"] = net.b_i.detach().clone()
    return out


def fold_in_columns(f: Dict[str, torch.Tensor], R: np.ndarray, C: np.ndarray, cols: np.ndarray,
                    cfg: FitConfig):
    """Label-blind vectors (and instance biases) for ``cols`` from their predictions alone."""
    X = f["X"]
    return fold_in(X, _t(R[:, cols], X.device, X.dtype), _t(C[:, cols], X.device, X.dtype),
                   cfg.lambda_reg, link=cfg.link, b_u=f.get("b_u"))


def R_hat_from_net(net, R, C, L, T, cfg: FitConfig, inductive: bool) -> np.ndarray:
    """R_hat (m x n) from a two-stage net.

    Inductive nets are fitted on the sorted labeled columns, ``np.sort(L)`` (the order
    ``two_stage_R_hat`` and ``label_row`` use); the test columns are folded in.
    """
    if not inductive:
        return net.reconstruct().detach().cpu().numpy()
    f = factors_of(net)
    R_hat = np.empty_like(R)
    R_hat[:, np.sort(L)] = net.reconstruct().detach().cpu().numpy()
    with torch.no_grad():
        Y_T, b_T = fold_in_columns(f, R, C, T, cfg)
        R_hat[:, T] = reconstruct(f["X"], Y_T, cfg.link, f.get("b_u"), b_T).cpu().numpy()
    return R_hat


def two_stage_R_hat(R, C, L, T, cfg: FitConfig, seed: int, inductive: bool) -> np.ndarray:
    """R_hat (m x n) of the confirmed factorization; inductive: fitted on L, T folded in."""
    cols = np.sort(L) if inductive else np.arange(R.shape[1])
    return R_hat_from_net(two_stage(R, C, cols, cfg, seed), R, C, L, T, cfg, inductive)


# --- option 1: the matched joint objective --------------------------------------------------------

def matched_joint(R, C, y, L, T, cfg: FitConfig, seed: int):
    """Joint fit with stop-gradient on labeled instance factors. Returns (g scores on T, R_hat)."""
    labels = np.full(R.shape[1], np.nan)
    labels[L] = y[L]
    tr = CFEnsemblePyTorchTrainer(
        n_classifiers=R.shape[0], latent_dim=cfg.d, rho=cfg.rho, lambda_reg=cfg.lambda_reg,
        aggregator_type="weighted", max_epochs=cfg.max_epochs, lr=cfg.lr, lr_g=cfg.lr_g,
        refit_g_steps=cfg.refit_g_steps, patience=cfg.patience, min_delta=cfg.min_delta,
        normalize_recon=True, sup_updates_instances=False, link=cfg.link, verbose=False,
        random_seed=seed, device=cfg.device,
    ).fit(EnsembleData(R, labels, C=C))
    return tr.predict()[T], tr.model.reconstruct().detach().cpu().numpy()


# --- options 2 and 3: the label row, with or without episodes -------------------------------------

class LabelRowCF(nn.Module):
    """CF factors for the classifier rows plus one label row (vector x_label, bias b_label)."""

    def __init__(self, f: Dict[str, torch.Tensor], link: str):
        super().__init__()
        self.link = link
        self.X = nn.Parameter(f["X"].clone())
        self.Y = nn.Parameter(f["Y"].clone())
        d = self.X.shape[0]
        if link == "logit":
            self.b_u = nn.Parameter(f["b_u"].clone())
            self.b_i = nn.Parameter(f["b_i"].clone())
        self.x_label = nn.Parameter(torch.zeros(d, dtype=self.X.dtype, device=self.X.device))
        self.b_label = nn.Parameter(torch.zeros((), dtype=self.X.dtype, device=self.X.device))

    def recon_loss(self, R: torch.Tensor, C: torch.Tensor, lam: float) -> torch.Tensor:
        """The trainer's normalized reconstruction + penalty over the fitted columns."""
        Z = self.X.T @ self.Y
        if self.link == "logit":
            entry = F.binary_cross_entropy_with_logits(Z + self.b_u + self.b_i, R, reduction="none")
            pen = (self.X ** 2).sum() + (self.Y ** 2).sum() + (self.b_i ** 2).sum()
        else:
            entry = (R - Z) ** 2
            pen = (self.X ** 2).sum() + (self.Y ** 2).sum()
        return ((C * entry).sum() + lam * pen) / R.numel()

    def label_logits(self, Y: torch.Tensor, b_i: Optional[torch.Tensor]) -> torch.Tensor:
        z = self.x_label @ Y + self.b_label
        return z + b_i.reshape(-1) if b_i is not None else z

    def free_label_logits(self, idx: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """Label-row logits from the free vectors; ``detach`` makes them label-blind."""
        Y = self.Y[:, idx]
        b = self.b_i[0, idx] if self.link == "logit" else None
        if detach:
            Y = Y.detach()
            b = b.detach() if b is not None else None
        return self.label_logits(Y, b)

    def factors(self) -> Dict[str, torch.Tensor]:
        out = {"X": self.X.detach(), "Y": self.Y.detach()}
        if self.link == "logit":
            out["b_u"], out["b_i"] = self.b_u.detach(), self.b_i.detach()
        return out


def _stratified_hide(y_L: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask over L: ``frac`` of each class hidden, at least one positive if any."""
    hide = np.zeros(len(y_L), dtype=bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y_L == cls)
        k = int(round(frac * len(idx)))
        if cls == 1 and len(idx) > 1:
            k = min(max(k, 1), len(idx) - 1)  # keep at least one positive visible too
        if k > 0:
            hide[rng.choice(idx, k, replace=False)] = True
    return hide


def label_row(R, C, y, L, T, cfg: FitConfig, seed: int, episodes: bool, inductive: bool,
              warm: Optional[Dict[str, torch.Tensor]] = None, blind: Optional[bool] = None):
    """Option 2 (episodes=False, blind=False), its stop-gradient form (episodes=False,
    blind=True) and option 3 (episodes=True; blind defaults to True).

    ``blind``: the label terms of non-hidden labeled instances use detached free vectors.

    Transductive: every column is in the reconstruction, test columns' vectors are free
    parameters fitted by reconstruction alone (label-blind), and a test score is read from
    them. Inductive: only the labeled columns are fitted; test columns are folded in at the
    end. ``warm`` is the two-stage fit on the same columns (computed if not given).

    Returns (scores on T, R_hat m x n, scores on L from the free vectors, epochs).
    """
    if blind is None:
        blind = episodes
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cols = np.sort(L) if inductive else np.arange(R.shape[1])
    if warm is None:
        warm = factors_of(two_stage(R, C, cols, cfg, seed))
    model = LabelRowCF(warm, cfg.link)
    dev, dt = model.X.device, model.X.dtype

    R_c, C_c = _t(R[:, cols], dev, dt), _t(C[:, cols], dev, dt)
    pos_in_cols = {c: k for k, c in enumerate(cols)}
    L_local = torch.as_tensor([pos_in_cols[i] for i in L], device=dev)  # L's positions in cols
    y_L = _t(y[L], dev, dt)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, ema, wait, epoch = np.inf, None, 0, 0
    for epoch in range(1, cfg.refine_epochs + 1):
        opt.zero_grad()
        rec = model.recon_loss(R_c, C_c, cfg.lambda_reg)
        if episodes:
            hide = torch.as_tensor(_stratified_hide(y[L], cfg.hide_frac, rng), device=dev)
            keep_idx, hide_idx = L_local[~hide], L_local[hide]
            Y_h, b_h = fold_in(model.X, R_c[:, hide_idx], C_c[:, hide_idx], cfg.lambda_reg,
                               link=cfg.link, b_u=getattr(model, "b_u", None))
            logits = torch.cat([model.free_label_logits(keep_idx, detach=blind),
                                model.label_logits(Y_h, b_h)])
            targets = torch.cat([y_L[~hide], y_L[hide]])
        else:
            logits, targets = model.free_label_logits(L_local, detach=blind), y_L
        sup = _balanced_bce(logits, targets) + cfg.lambda_reg * (model.x_label ** 2).sum() / len(L)
        loss = cfg.rho * rec + (1 - cfg.rho) * sup
        loss.backward()
        opt.step()
        val = loss.item()
        ema = val if ema is None else 0.9 * ema + 0.1 * val  # episodes make the loss noisy
        if ema < best - cfg.min_delta:
            best, wait = ema, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    f = model.factors()
    with torch.no_grad():
        scores_L = torch.sigmoid(model.free_label_logits(L_local)).cpu().numpy()
        R_hat = np.empty_like(R)
        R_hat[:, cols] = reconstruct(f["X"], f["Y"], cfg.link, f.get("b_u"), f.get("b_i")).cpu().numpy()
        if inductive:
            Y_T, b_T = fold_in_columns(f, R, C, T, cfg)
            R_hat[:, T] = reconstruct(f["X"], Y_T, cfg.link, f.get("b_u"), b_T).cpu().numpy()
            scores_T = torch.sigmoid(model.label_logits(Y_T, b_T)).cpu().numpy()
        else:
            T_local = torch.as_tensor(T, device=dev)  # cols is every column
            scores_T = torch.sigmoid(model.free_label_logits(T_local)).cpu().numpy()
    return scores_T, R_hat, scores_L, epoch
