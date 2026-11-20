# Round 6 — ranking loss and alternating CF-selection

*Part of [the results ledger](index.md), which holds the standing verdict, the rules and the settings every round shares. Previous: [round 5](round-05-joint-simplex.md) · Per-seed results: [protein_homo](reports/round-06-protein_homo.csv), [coffin_splice](reports/round-06-coffin_splice.csv).*

**Run:** 2026-10-21/22 on a RunPod A40 · **Script:** `05_ranking_and_alternating.py` · **Exploratory**
(dev seeds: COFFIN 0–4, protein_homo 0–2) · **Output:** `output/benchmarks/<dataset>/ranking_alt_base0/`

## Question

[Round 5](round-05-joint-simplex.md) blamed the in-model simplex combiner's failure on its cross-entropy objective. Does a ranking
objective fix it, and does alternating greedy selection with ranking-steered factorization beat
post-hoc selection on $\hat{R}$?

- **Ranking loss:** `sup_loss='pairwise'`, mean $\log(1 + e^{-\kappa (s_{\text{pos}} - s_{\text{neg}})})$
  over positive/negative pairs of combiner scores (κ = 20 for the simplex, 1 for logits).
- **`alt_select`** (`alternating.py`): two-stage init; 5 rounds of [greedy ensemble selection on
  $\hat{R}_ L$ → 200 epochs of $\rho L_{\text{recon}} + (1-\rho) L_{\text{rank}}$ with α frozen and
  stop-gradient on labeled instances]; final selection on $\hat{R}_ L$.

## Results (PR-AUC on T, mean over seeds; logit link)

**COFFIN** (classifier confidence, d=8):

| Method | 2% | 5% | 20% | Mean |
|---|---|---|---|---|
| ens_select on raw R | 0.276 | 0.291 | 0.297 | **0.288** |
| cf_ens_select (post hoc on $\hat{R}$) | 0.245 | **0.301** | **0.300** | 0.282 |
| **alt_select, ρ=0.9** | **0.270** | 0.281 | 0.283 | 0.278 |
| alt_select, ρ=0.5 | 0.250 | 0.273 | 0.279 | 0.268 |
| stacking | 0.253 | 0.268 | 0.275 | 0.265 |
| joint_simplex_rank, ρ=0.5 | 0.232 | 0.252 | 0.251 | 0.245 |
| two_stage_simplex_rank | 0.228 | 0.248 | 0.256 | 0.244 |
| joint_lr_rank, ρ=0.9 | 0.259 | 0.210 | 0.170 | 0.213 |

**protein_homo:** cf_ens_select (learned, d=4) 0.895 · alt_select (classifier, d=8) 0.893 ·
cf_ens_select (classifier, d=8) 0.892 · ens_select 0.891 · joint_simplex_rank 0.888–0.889 · stacking 0.888.

Paired: alt_select ρ=0.9 vs. cf_ens_select on COFFIN −0.004 (n.s., p = 0.28); vs. raw ens_select −0.010
(p = 0.04). On protein_homo alt_select ties cf_ens_select (+0.0004) and is +0.002 over raw ens_select (n.s.).

## Findings

1. **A ranking objective does not rescue the in-model simplex.** Jointly trained it is unchanged
   (0.245 vs. 0.246 with cross-entropy, [round 5](round-05-joint-simplex.md)). It does help the two-stage simplex (+0.020), but that
   still trails greedy selection by about 0.04. The issue is not only the loss: smooth softmax weights
   fitted on 10–104 positives are less robust than greedy, discrete PR-AUC selection.
2. **Alternating CF–selection is the best in-loop method and fixes the low-budget weakness.** At 2% on
   COFFIN, where post-hoc selection on $\hat{R}$ is unstable (0.245), alt_select reaches 0.270. It gives
   that back at 5% and 20%.
3. **Overall, label-steered factorization still does not beat two-stage CF + post-hoc selection**, the
   method confirmed in [round 4](round-04-confirmation.md).

## Implications

- The confirmed method stands: logit-link factorization with classifier and instance biases and a
  confidence matrix, two-stage, followed by ensemble selection on $\hat{R}$.
- A budget-aware hybrid is suggested by finding 2: alt_select when there are very few labeled positives,
  post-hoc selection otherwise. That is a hypothesis to test on fresh seeds, not a result.
- Base-classifier seeds 1 and 2 were generated in this run (`base_seed1`, `base_seed2`) for the next
  confirmation, which should also choose the confidence type on L by cross-validation.
