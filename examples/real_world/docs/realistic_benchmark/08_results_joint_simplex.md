# Results: Selection Built Into CF (Joint Simplex Combiner)

**Run:** 2026-10-21 on a RunPod A40 · **Script:** `04_joint_simplex.py` · **Exploratory** (dev seeds:
COFFIN 0–4, protein_homo 0–2) · **Output:** `output/benchmarks/<dataset>/joint_simplex_base0/`

## Question

Docs 05 and 07 show that CF helps when ensemble selection is applied to $\hat{R}$ *after* the
factorization. Can the selection bias live inside the model instead? The combiner becomes
$g(\hat{r}_ {\cdot i}) = \sum_u \alpha_u \hat{r}_ {ui}$ with $\alpha = \text{softmax}(\theta)$: nonnegative,
summing to one. Training uses stop-gradient (supervision updates $X$, $b_u$ and $\alpha$, never the
labeled $y_i$), a separate learning rate for $g$, and a final refit of $g$ on the frozen reconstruction.

## Results (PR-AUC on T, mean over seeds; logit link, fixed configurations)

**COFFIN** (classifier confidence, d=8):

| Method | 2% | 5% | 20% | Mean |
|---|---|---|---|---|
| ens_select on raw R | 0.276 | 0.291 | 0.297 | **0.288** |
| cf_ens_select (post hoc on $\hat{R}$) | 0.245 | 0.301 | 0.300 | 0.282 |
| stacking on raw R | 0.253 | 0.268 | 0.275 | 0.265 |
| joint_simplex_sg, ρ=0.5 | 0.241 | 0.252 | 0.247 | 0.246 |
| joint_simplex_sg, ρ=0.9 | 0.210 | 0.246 | 0.238 | 0.231 |
| two_stage_simplex (ρ=1) | 0.227 | 0.223 | 0.221 | 0.224 |
| joint_lr_sg, ρ=0.9 (logistic $g$, refit) | 0.181 | 0.161 | 0.185 | 0.176 |

**protein_homo** (classifier confidence, d=8): cf_ens_select 0.892 · ens_select 0.891 · joint_simplex_sg
ρ=0.5 0.883 · two_stage_simplex 0.872 · joint_lr_sg 0.855.

Every in-model combiner is significantly worse than post-hoc ensemble selection on $\hat{R}$
(COFFIN: −0.036 to −0.106, 1–2 of 15 cells won).

## Diagnosis

1. **The simplex combiner selects the wrong models.** On COFFIN its three largest weights are most
   often AdaBoost, **Gaussian NB (base PR-AUC 0.04)** and LDA or the **decision tree (0.01)**. The
   training objective, class-weighted cross-entropy of an *average of probabilities*, rewards models that
   output extreme probabilities on the few labeled positives, which is what Naive Bayes does. Ensemble
   selection instead maximizes PR-AUC, a *ranking* metric, directly on the labeled set.
2. **Joint beats its own two-stage counterpart** (ρ=0.5 vs. ρ=1: +0.022 on COFFIN, +0.012 on protein_homo).
   Letting supervision shape the classifier factors and biases does help; it just cannot make up for the
   wrong combiner objective.
3. **Fully training the logistic $g$ overfits.** With the refit, the joint logistic combiner is now
   converged but unregularized (this script did not set `lambda_g`), and 10–104 labeled positives are
   not enough for an unregularized fit; it falls below the plain average.

So the problem is not the selection *bias* (nonnegative weights summing to one) but the *objective* used
to fit it. In-model selection needs a ranking objective.

## Next steps

1. **Ranking loss for the in-model combiner:** fit $\alpha$ (and, jointly, $X$ and $b_u$) with a pairwise
   AUC surrogate, $\sum_{i \in L^+, j \in L^-} \log(1 + e^{-(s_i - s_j)})$ with $s = \alpha^\top \hat{r}$.
   It handles imbalance naturally (pairs, not instances) and targets ordering, like ensemble selection.
2. **Alternating CF–selection:** alternate a greedy ensemble-selection step on $\hat{R}_ L$ with a
   factorization step whose supervised term scores the current selection with the ranking loss. This
   keeps the greedy selector that works, while still letting labels steer $X$ and $b_u$.
3. Always pair a fully trained combiner with regularization (`lambda_g`, or a fixed step budget).
