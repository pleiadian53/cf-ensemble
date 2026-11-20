# Round 3 — confidence weighting, logistic link and combiners on R̂

*Part of [the results ledger](index.md), which holds the standing verdict, the rules and the settings every round shares. Previous: [round 2](round-02-coffin-splice.md) · Next: [round 4](round-04-confirmation.md) · Per-seed results: [protein_homo](reports/round-03-protein_homo.csv), [coffin_splice](reports/round-03-coffin_splice.csv).*

**Run:** 2026-10-21 on a RunPod A40 · **Script:** `03_confidence_and_link.py` ·
**Output:** `output/benchmarks/<dataset>/confidence_link_base0/` · COFFIN 5 seeds, protein_homo 3 seeds

## Question

COFFIN showed Caruana ensemble selection winning because it drops useless base models, while a
low-rank $\hat{R}$ mixes every model into every reconstructed row. Can the CF representation do the
same if (a) the confidence matrix $C$ tells the factorization which entries to trust, (b) a logistic
link with classifier and instance biases models the probabilities better, and (c) a selection-style
combiner is applied to $\hat{R}$?

All factorizations are two-stage ($\rho = 1$); labels only enter through $C$ (classifier and learned
confidence), never the reconstruction target.

| Factor | Levels |
|---|---|
| Confidence $C$ | uniform · classifier ($q_u \propto \max(\text{AUC}_ u - 0.5, 0)$ on L) · learned (cell-level reliability, class-balanced, cross-fitted) |
| Link | identity ($\hat{R} = X^\top Y$) · logit ($\hat{R} = \sigma(X^\top Y + b_u + b_i)$, cross-entropy) |
| Latent dim | 4, 8 |
| Combiner on $\hat{R}$ | logistic regression (`cf_lr`) · ensemble selection (`cf_ens_select`) |

## COFFIN (PR-AUC on T, mean over 5 seeds, fixed configurations)

| Method | 2% (10 pos) | 5% | 20% | Mean |
|---|---|---|---|---|
| **ens_select on raw R** | **0.276** | 0.291 | 0.297 | **0.288** |
| cf_ens_select · classifier · identity · d=8 | 0.262 | 0.295 | 0.296 | 0.284 |
| cf_ens_select · classifier · logit · d=8 | 0.245 | **0.301** | **0.300** | 0.282 |
| cf_ens_select · uniform · logit · d=4 | 0.265 | 0.289 | 0.291 | 0.282 |
| cf_lr · classifier · logit · d=8 | 0.257 | 0.271 | 0.277 | 0.268 |
| stacking on raw R | 0.253 | 0.268 | 0.275 | 0.265 |
| two_stage ([round 2](round-02-coffin-splice.md): uniform · identity · LR) | 0.251 | 0.265 | 0.268 | 0.261 |

Paired against raw `ens_select` over all 15 seed × budget cells, **no CF configuration is
significantly better**. The best (classifier · identity · d=8) is −0.004 (Wilcoxon p = 0.49); the
logit variant wins at 5% and 20% but loses badly at 2%. Learned cell-level confidence is
significantly *worse* (−0.013 to −0.020, p ≤ 0.002).

Averages over all `cf_ens_select` configurations, by factor:

| Factor | Levels (mean PR-AUC) |
|---|---|
| Link | identity 0.263 → **logit 0.278** |
| Confidence | learned 0.264 · uniform 0.270 · **classifier 0.276** |
| Latent dim | 4: 0.265 → **8: 0.275** |

## protein_homo (mean over 3 seeds)

| Method | 2% | 5% | 20% | Mean |
|---|---|---|---|---|
| **cf_ens_select · learned · logit · d=4** | **0.893** | **0.898** | 0.895 | **0.895** |
| cf_ens_select · uniform · logit · d=4 | 0.889 | 0.895 | **0.896** | 0.893 |
| ens_select on raw R | 0.888 | 0.892 | 0.893 | 0.891 |
| stacking on raw R | 0.882 | 0.892 | 0.890 | 0.888 |

`cf_ens_select · learned · logit · d=4` beats raw ensemble selection in **all 9 cells** (+0.005,
Wilcoxon p = 0.004, the smallest attainable with 9 pairs). Because about 24 configurations were
compared, this is **a hypothesis to confirm on fresh seeds**, not yet a result. Logit again beats
identity (0.893 vs 0.889 averaged over configurations).

## What we learned

1. **The combiner matters most.** Moving from logistic regression to ensemble selection on $\hat{R}$
   lifts CF on COFFIN from 0.261–0.268 to 0.28: the same representation, a better inductive bias.
2. **The logistic link with biases helps on both datasets** (COFFIN +0.015 averaged over
   configurations). Probabilities are better modeled in logit space, and $b_u$ can absorb a
   classifier's systematic shift.
3. **Per-classifier confidence helps on COFFIN** (+0.006 over uniform), which supports the idea that
   telling the factorization which models to trust is useful. Learned *cell-level* confidence hurts on
   COFFIN (10–104 positives, weak base models: too little signal to learn per-cell reliability) but
   gives the best configuration on protein_homo (13–130 positives, strong base models).
4. **The 2% budget is where CF loses on COFFIN.** With 10 positives, per-classifier quality and the
   selection on $\hat{R}$ are both noisy; raw ensemble selection is more robust there.
5. CF has moved from "plain-average level" ([round 2](round-02-coffin-splice.md)) to **parity with the strongest baseline**, with a
   possible small win where base models are strong. It does not yet beat ensemble selection robustly.

## Next steps

1. **Confirmation run (pre-registered):** `cf_ens_select · learned · logit · d=4` and
   `cf_ens_select · classifier · logit · d=8` against raw `ens_select` on fresh seeds (5–14), both datasets.
2. **Shrink confidence estimates at low budgets:** pull $q_u$ toward uniform with a prior whose weight
   falls as the number of labeled positives grows; combine $q_u$ with cell reliability
   ($c_{ui} = q_u \cdot \text{rel}_ {ui}$).
3. **Build the selection bias into CF itself:** replace the post-hoc combiner with nonnegative simplex
   weights (softmax) on $\hat{R}$, trained with the factorization once the joint optimizer is fixed
   (separate optimizer and early stopping for $g$).
