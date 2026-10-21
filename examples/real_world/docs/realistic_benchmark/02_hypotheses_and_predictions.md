# Hypotheses and Pre-Registered Predictions

Written on 2026-10-20, before any results. Results docs will score these predictions.

## When should the joint objective help?

The joint loss combines three jobs:

- **Reconstruction uses unlabeled data.** Every instance shows how the classifiers agree and
  disagree. A low-rank $\hat{R}$ keeps the shared structure in those patterns (for example,
  "these three classifiers fail together on this kind of instance") and discards the rest as noise.
- **Supervision chooses which structure to keep.** Without it, the factorization keeps whatever
  explains the most variance in $R$, not what predicts $y$. This is the idea behind supervised PCA
  and reduced-rank regression.
- **Joint learning of $g$** lets the subspace and the combiner adapt to each other.

### Where the ceiling is

With a dot-product interaction and a logistic-regression $g$, two-stage is roughly "project
each column of $R$ onto a learned low-dimensional subspace, then run logistic regression": a
restricted form of stacking. Its advantage is *lower variance* when labels are scarce, not more
expressive power. As labels grow, cross-validated stacking (the Super Learner) is provably
close to the best weighted combination of the base models, so CF-Ensemble should not be
expected to beat it in the many-label regime.

### The niche where it could win

- few labels, many unlabeled instances;
- transductive: test instances available during fitting;
- base classifiers with *structured* rather than independent errors.

Genomics fits this unusually well: labels are expensive, but scoring millions of unlabeled
positions is free.

### Positioning in the literature

CF-Ensemble sits between two families it must be compared with:

- **Supervised combiners:** stacking and the Super Learner; Caruana et al. (2004) greedy
  ensemble selection, a strong practical baseline.
- **Label-free combiners that also exploit low-rank structure in $R$:** the Spectral
  Meta-Learner (Parisi et al. 2014, PNAS); Dawid–Skene and Snorkel's label model.

A convincing result is a win over both families in the few-label regime.

## A predicted failure mode: labeled-point distortion

The supervised gradient reaches the instance factors $y_i$ of **labeled** points, each with
$d$ free parameters but only $m$ observations. Those factors can bend until their
reconstructed columns produce the correct label. Test points' factors only feel the
reconstruction pull and stay undistorted. So $g$ is fitted on distorted columns and applied to
undistorted ones: a train/test mismatch built into the transductive objective.

Supervision shaping the shared classifier factors $X$ (the useful part) is a different matter
from supervision bending individual labeled $y_i$ (the leaky part). Two ways to separate them:

1. **Stop-gradient:** supervision updates $X$ and $g$, but not the labeled $y_i$.
2. **Amortized instance factors:** $y_i = f(\text{features}_ i)$ with $f$ shared, so no
   instance has private parameters to bend.

## Beyond the dot product

The interaction $\hat{r}_ {ui} = h(x_u, y_i)$ can be any function, but a flexible $h$ on top of an
instance factor with about 11 observations just memorizes the column and removes the denoising.
Rendle et al. (2020) found a well-tuned dot product beats MLP interactions in collaborative
filtering. Options in order of promise:

1. **Logistic link with biases:** $\hat{r}_ {ui} = \sigma(x_u^\top y_i + b_u + b_i)$, with
   cross-entropy reconstruction. $b_u$ captures a classifier's over- or under-prediction, $b_i$ an
   instance's inherent positivity. Close to item-response theory.
2. **Per-classifier calibration:** $\hat{r}_ {ui} = \varphi_u(x_u^\top y_i)$ with $\varphi_u$
   monotone. Base classifiers differ strongly in calibration; each $\varphi_u$ has $n$ points to learn from.
3. **Amortized instance factors**, as above; also enables inductive prediction.
4. **Free-form MLP or metric interactions**, only with strong regularization.

## Pre-registered predictions

### Smoke test: `protein_homo`, m = 11, d = 20 (no rank bottleneck), budgets 2/5/20%, 3 seeds

1. Gradient boosting, random forest, extra trees, logistic regression and MLP lead (PR-AUC
   about 0.8–0.9); Gaussian NB is worst.
2. `two_stage` ≈ `stacking` (difference below 0.01 at every budget): with $d \ge m$, $\hat{R}$ is
   a shrunk copy of $R$.
3. `joint_sum_rho0.5` ≈ `two_stage`: the summed reconstruction swamps supervision for $X, Y$,
   but Adam rescales per parameter, so $w, b$ still train normally.
4. `joint_rho0.1` is the worst learned combiner, especially at 2% (labeled-point distortion);
   `joint_rho0.9` ≈ `two_stage`.
5. At 2% (about 13 positives) `mean` / `best_single` match or beat `stacking`; at 20%,
   `stacking` ≥ `best_single` ≥ `mean`.
6. The top methods land within about one seed-std of each other: no significant CF win.

### Headline: `coffin_splice`, d < m

- Base models reach PR-AUC about 0.3–0.5 (below the published 0.5–0.58, which used string-kernel SVMs).
- CF variants within ±0.02 of stacking. Probability of a clear win at the smallest budget:
  25–35%; at 20%: below 10%.
- The default certainty confidence $c_{ui} = \vert r_{ui} - 0.5 \vert$ is likely
  counterproductive under heavy imbalance: it gives the most weight to easy negatives near 0
  and the least to the uncertain entries around the positives.

### What would change the outlook

- If `joint` beats `two_stage` at low budgets even with $d = 20$, supervision is shaping $X$
  usefully despite the distortion: a good sign for the method.
- If prediction 4 holds, the stop-gradient variant is the natural next step.
