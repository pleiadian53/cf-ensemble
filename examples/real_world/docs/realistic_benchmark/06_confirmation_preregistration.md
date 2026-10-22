# Pre-Registration: Confirmation Run for CF + Ensemble Selection

**Written and committed 2026-10-21, before the run.** Results will go in a separate doc and are
scored against this one without changes.

## Why

Doc 05 compared about 24 CF configurations against raw ensemble selection on the same seeds that
were used to pick the best ones. Two configurations stood out; this run tests them on fresh seeds
so the selection cannot flatter them.

## Primary hypotheses (confirmatory)

| ID | Dataset | Configuration | Claim |
|---|---|---|---|
| **H1** | protein_homo | `cf_ens_select` · learned · logit · d=4 | PR-AUC > raw `ens_select` |
| **H2** | coffin_splice | `cf_ens_select` · classifier · logit · d=8 | PR-AUC > raw `ens_select` |

Both configurations also run on the other dataset; those cross results are reported but are
secondary.

## Protocol

- **Fresh seeds 5–14** (10 seeds) × label budgets 2%, 5%, 20% → **30 paired cells per dataset**.
  Seeds 0–4 (used in doc 05) are excluded.
- Same probability matrices R as before (`base_seed0`). Seeds vary the labeled/test split and the
  factorization initialization; the base classifiers are not retrained (a known limitation).
- Everything else as in doc 05: two-stage factorization (ρ = 1), max 3,000 epochs, λ = 0.01,
  ensemble selection with 50 greedy steps.

## Decision rule

- Per hypothesis: **one-sided Wilcoxon signed-rank test** on the 30 paired differences
  (CF minus raw `ens_select`), H₁: median difference > 0.
- **Holm correction** across H1 and H2 at α = 0.05.
- A hypothesis is **confirmed** if its corrected p < 0.05 **and** the mean difference is positive.
  Mean difference, win count, and per-budget means are reported regardless.

## Exploratory (not confirmatory)

Shrinkage of per-classifier quality toward uniform, $\tilde{q}_ u = \lambda q_u + (1 - \lambda)$ with
$\lambda = n_{\text{pos}} / (n_{\text{pos}} + 20)$, and the combined confidence
$c_{ui} = \tilde{q}_ u \cdot \text{rel}_ {ui}$, each with the logit link at d = 4 and d = 8.
Expected to reduce the losses at the 2% budget; any win here needs its own confirmation.

## Predictions (my own, before the run)

- H1 confirmed: about 45%. The doc-05 effect (+0.005, 9/9 cells) is small, and some of it is
  likely selection luck.
- H2 confirmed: about 20%. In doc 05 it lost badly at 2% (−0.030), which a one-sided test over all
  budgets is unlikely to overcome.
- Shrunk classifier confidence beats unshrunk classifier confidence at 2% on COFFIN.
