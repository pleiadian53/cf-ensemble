# Results: COFFIN Human Acceptor Splice Sites

**Run:** 2026-10-21 on a RunPod A40 (no cheaper GPU was available) · **Output:**
`output/benchmarks/coffin_splice/` · **Seeds:** 5 per label budget

## Data

- 1M-example COFFIN training subset → stratified 300,000 rows, **0.346% positive** (about 1:289).
- Features: one-hot over the 141-nt window plus 3-mer counts of each half (692 features).
- D_base 150,000 rows (520 positives) · D_ens 150,000 rows (519 positives).
- At the 2% budget the combiners see **10 labeled positives**; at 20%, 104.

## Base classifiers (PR-AUC on D_ens)

| Model | PR-AUC | ROC-AUC | Fit time |
|---|---|---|---|
| linear_svm | 0.271 | 0.963 | 24 s |
| logreg | 0.263 | 0.964 | 4 s |
| mlp | 0.189 | 0.946 | 10 s |
| adaboost | 0.189 | 0.951 | 287 s |
| lda | 0.177 | 0.954 | 10 s |
| extra_trees | 0.136 | 0.931 | 21 s |
| hist_gbm | 0.119 | 0.955 | 75 s |
| random_forest | 0.113 | 0.914 | 17 s |
| gaussian_nb | 0.043 | 0.923 | 1 s |
| knn | 0.021 | 0.654 | 1 s |
| decision_tree | 0.011 | 0.579 | 12 s |

Weak and very uneven: the published state of the art (string-kernel SVMs on all 50M examples)
reaches PR-AUC about 0.5–0.58. Three models are close to useless.

## Ensemble methods (PR-AUC on T, mean over 5 seeds, fixed configurations)

| Method | 2% (10 pos) | 5% (26 pos) | 20% (104 pos) | Mean |
|---|---|---|---|---|
| **ens_select (Caruana)** | **0.276** | **0.291** | **0.297** | **0.288** |
| stacking | 0.253 | 0.268 | 0.275 | 0.265 |
| two_stage, d=8 | 0.251 | 0.265 | 0.268 | 0.261 |
| best joint / joint_sg setting (joint, d=8, ρ=0.5) | 0.260 | 0.262 | 0.256 | 0.259 |
| mean (no labels) | 0.260 | 0.261 | 0.254 | 0.258 |
| best_single | 0.248 | 0.233 | 0.261 | 0.247 |
| summa (no labels) | 0.213 | 0.213 | 0.208 | 0.211 |

Paired over seeds:

- **ens_select beats two_stage in every seed and budget**, by 0.025–0.028 on average per budget.
- **two_stage trails stacking in every seed and budget** (0/5), by 0.002–0.007 on average: small but consistent.
- **Every joint and joint_sg configuration lands at the level of `mean`.**

## Why joint training fails here (diagnostic, seed 0, 5% labels, d=8)

| Variant | Joint $g$ PR-AUC | scikit-learn LR refit on that run's $\hat{R}$ | Largest $\vert w \vert$ (joint / refit) |
|---|---|---|---|
| stacking (reference) | — | 0.280 | — / 10.5 |
| joint, ρ=0.1 / 0.5 / 0.9 | 0.260 / 0.261 / 0.259 | 0.260 / 0.259 / 0.257 | 0.5–4.3 / 0.6–4.2 |
| joint_sg, ρ=0.1 / 0.5 / 0.9 | 0.262 / 0.261 / 0.259 | 0.271 / 0.271 / 0.274 | 2.6–3.1 / 7.5–9.0 |

Two separate failures:

1. **Without stop-gradient, the labeled columns are corrupted.** Even a refit combiner cannot learn
   anything transferable from them: labeled instance factors bend toward their labels, so
   $\hat{R}$ on L no longer resembles $\hat{R}$ on T.
2. **With stop-gradient, the joint $g$ is under-trained.** Early stopping and the learning-rate
   schedule watch the *total* loss, which reconstruction dominates; by the time reconstruction
   converges, $g$'s weights are about a third of the converged scikit-learn solution, and it still
   ranks almost like the plain average. Refitting $g$ properly recovers about +0.01.

Even the repaired best case (stop-gradient plus a properly fitted $g$, 0.271–0.274) stays below
stacking on raw $R$ (0.280). On this data, letting labels steer the factorization did not
produce a better $\hat{R}$ than two-stage.

The L2 penalty on $g$ (`lambda_g` = 1/(2|L|), matching scikit-learn's C=1) made no measurable
difference on `protein_homo` (rerun tagged `gl2`): at |L| ≈ 1,450 the penalty is tiny, so missing
regularization was not the bottleneck there.

## What the winner suggests

Caruana ensemble selection wins because of its **inductive bias**: nonnegative, sparse, convex
weights chosen greedily. With 10–100 labeled positives and three nearly useless base models, that
bias drops the bad models and averages the good ones, while logistic stacking (and any combiner on
a low-rank $\hat{R}$, which mixes every model into every reconstructed row) has to learn signed
weights for all 11 inputs. The low-rank denoising hypothesis is **not supported** on this data.

SUMMA (label-free, spectral) is the worst method: its conditional-independence assumption fails
for 11 strongly correlated classifiers.

## Scoring the COFFIN predictions

| Prediction | Outcome |
|---|---|
| Base models reach PR-AUC about 0.3–0.5 | ❌ lower: best 0.27, several near 0 |
| CF within ±0.02 of stacking | ✅ two_stage −0.002 to −0.007; joint −0.007 to −0.02 |
| Clear CF win at the smallest budget: 25–35% | ✅ consistent: no win |
| Clear CF win at 20%: below 10% | ✅ no win |

## Options from here

1. **Give the CF representation the winning combiner:** run ensemble selection (or nonnegative
   weights) on $\hat{R}$ instead of $R$. That tests denoising separately from the combiner's
   inductive bias. Cheap: reuses the saved $R$.
2. **Fix the joint optimizer:** a separate optimizer and learning rate for $g$, or a convex refit of
   $g$ after each factorization step, with early stopping on the supervised term. Upper bound from
   the diagnostic: roughly two-stage level.
3. **Change the model, per the hypotheses doc:** logistic link with classifier and instance biases
   (its $b_u$ can absorb a useless classifier), and amortized instance factors from features.
4. **Strengthen the base layer:** the COFFIN base models are far below the published state of the art;
   a stronger, more diverse layer (for example, a k-mer string-kernel SVM) is closer to the regime
   the method was designed for.
