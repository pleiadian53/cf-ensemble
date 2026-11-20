# The results ledger

*An append-only record of every experimental round of the realistic benchmark for CF-Ensemble. This page holds what the rounds add up to: the standing verdict, the rules every round follows, the settings they share and what comes next. Each round has its own page, which says what was asked, how it was run, what was measured and what was concluded, and links to the per-seed results it was computed from.*

> **What this ledger covers.** CF-Ensemble combines the predictions of several base classifiers by first transforming the matrix of their predicted probabilities with collaborative filtering, the matrix factorization behind recommender systems, and then combining the transformed predictions. This ledger tests that idea on real, noisy and highly imbalanced data, on two datasets, against the combiners a practitioner would otherwise use: the plain average, the best single model, stacking (a logistic regression on the predictions) and greedy ensemble selection. Earlier synthetic experiments are not part of it, because on those data the plain average already scored perfectly, so they could not separate one method from another.

---

## The rounds

| Round | Question | Answer | Page |
|---|---|---|---|
| 1 | Does the pipeline work, and do factorizing first and stacking afterwards (two-stage) or letting the labels steer the factorization (joint) beat stacking on protein_homo? | The pipeline works, and nothing separates the methods: every combiner lands between 0.876 and 0.893 PR-AUC, within seed-to-seed noise. Joint training collapses to the plain average, because the labeled instances' factors bend toward their own labels. Uniform confidence weights beat certainty weights. | [smoke test on protein_homo](round-01-protein-homo-smoke.md) |
| 2 | On the hard, extremely imbalanced COFFIN splice-site data, does CF beat the standard combiners? | No. Ensemble selection wins (0.288), ahead of stacking (0.265) and two-stage CF (0.261); joint CF lands at the plain average (0.259 against 0.258). Joint training fails twice over: without a stop-gradient the labeled columns are corrupted, and with one the combiner is under-trained. | [COFFIN splice sites](round-02-coffin-splice.md) |
| 3 | Can confidence weighting and a logistic link make the factorization useful, and which combiner should read it? | Yes, with ensemble selection as the combiner on the reconstructed matrix. CF reaches parity with raw ensemble selection on COFFIN (0.284 against 0.288) and beats it on protein_homo (0.895 against 0.891, all 9 cells). The logistic link adds about 0.015 on COFFIN. Exploratory: about 24 configurations compared on the same seeds. | [confidence, link and combiner](round-03-confidence-and-link.md) |
| 4 | Do round 3's two best configurations beat raw ensemble selection on fresh seeds, under a rule fixed in advance? | Yes, on both datasets, by small margins: +0.0033 PR-AUC on protein_homo (Holm-adjusted p = 0.0098) and +0.0046 on COFFIN (p = 0.0145; 0.286 against 0.282). The best confidence type depends on the dataset. | [pre-registered confirmation](round-04-confirmation.md) |
| 5 | Can the selection that makes ensemble selection strong be built into CF itself, as a convex combiner trained jointly? | No. The in-model combiner scores 0.246 on COFFIN, below stacking: cross-entropy on averaged probabilities favors the base models with the most extreme probabilities, which are among the weakest. | [joint simplex combiner](round-05-joint-simplex.md) |
| 6 | Does a ranking loss rescue the in-model combiner, and does alternating between CF and greedy selection help? | The ranking loss does not (0.245). Alternating CF and selection fixes COFFIN's 2% budget (0.270 against 0.245) but is 0.004 behind the round 4 method overall, not significant. Round 4's method stands. | [ranking loss and alternation](round-06-ranking-and-alternating.md) |

Rounds are numbered in the order they were run. Two companion pages hold what was written before a run: [round 1a](round-01a-preregistration.md), the hypotheses and predictions for rounds 1 and 2, and [round 4a](round-04a-preregistration.md), the pre-registration of round 4. The design shared by every round is on [its own page](experimental-design.md).

---

## The standing verdict

**A two-stage CF representation, read by ensemble selection, beats ensemble selection on the raw predictions, by a small margin.** The method is: factorize the prediction matrix with a logistic link and classifier and instance biases, $\hat{r}_ {ui} = \sigma(x_u^\top y_i + b_u + b_i)$, without labels in the reconstruction target ($\rho = 1$); let the labels enter only through the confidence weights $c_{ui}$ that say how far to trust each prediction; then run greedy ensemble selection on $\hat{R}$ instead of on $R$. Here $r_{ui}$ is base classifier $u$'s predicted probability for instance $i$, $x_u$ and $y_i$ are the latent vectors of classifier $u$ and instance $i$, $b_u$ and $b_i$ their biases, and $\sigma$ the logistic function. On fresh seeds, under a rule fixed before the run, this beat raw ensemble selection on both datasets ([round 4](round-04-confirmation.md)). The gains are about 0.003 to 0.005 PR-AUC, 1 to 2% relative.

**Letting the labels steer the factorization has not paid off so far.** Every joint variant tried, with or without a stop-gradient on the labeled instance factors and with logistic, convex or ranking combiners trained inside the model, lost to the two-stage method ([rounds 1, 2, 5 and 6](#the-rounds)). The failures are understood in part: a labeled instance's factor can absorb its own label, so the combiner is trained on representations unlike those it is applied to.

**Three limits travel with the result.** Every round so far reuses one set of base classifiers, trained on one split, so the gain has not been shown to survive retraining them. The confidence type was chosen per dataset (per-classifier quality on COFFIN, a learned cell-level reliability model on protein_homo), and a usable method must choose it from the labeled data alone. And the base classifiers on COFFIN are weak, about 0.29 PR-AUC against about 0.5 to 0.58 published for this task, so the gain is measured over weak, correlated models.

**Raw ensemble selection remains the reference** every CF configuration is compared with.

---

## The rules every round follows

- **Configurations are fixed before a run, and test labels are never used to choose them.** Where a round compares many configurations on the same seeds ([round 3](round-03-confidence-and-link.md)), its winners count as candidates only, and are re-tested on fresh seeds before any claim ([round 4](round-04-confirmation.md)).
- **Predictions and decision rules are written down first.** Rounds 1 and 2 were preceded by written predictions ([round 1a](round-01a-preregistration.md)), scored on their pages. Confirmatory rounds are pre-registered and committed before the run, with hypotheses, seeds, contrasts and the decision rule ([round 4a](round-04a-preregistration.md)).
- **Comparisons are paired.** Every method is scored on the same splits. A cell is one label budget under one seed, and methods are compared through per-cell differences in PR-AUC.
- **Confirmatory tests** are one-sided Wilcoxon signed-rank tests over the paired cells, with Holm's correction across the primary hypotheses. Exploratory comparisons are reported with the same arithmetic and carry no verdict.
- **The strongest baseline sets the bar.** Ensemble selection on the raw predictions, which beat stacking and every early CF variant, is the comparison that decides; beating the plain average or stacking is not enough.

---

## The settings every round shares

The full design is on [its own page](experimental-design.md). In brief:

- **Datasets.** `protein_homo`, KDD Cup 2004 protein homology: 145,751 rows, about 0.9% positive, 74 tabular features; the pipeline's smoke test. `coffin_splice`, human acceptor splice sites from Sonnenburg and Franc (2010): a stratified 300,000-row sample of the 1M-example training set, 0.346% positive, 141-nucleotide DNA windows encoded as one-hot and 3-mer counts (692 features); the headline benchmark.
- **Two-level split.** Each dataset is split in half. The base classifiers are trained on one half, $D_{\mathrm{base}}$, and predict the other, $D_{\mathrm{ens}}$, giving the prediction matrix $R$ of size $m \times n$, with $m$ base classifiers and $n$ instances, whose errors are honest out-of-sample errors. Within $D_{\mathrm{ens}}$, each seed draws a stratified labeled set $L$ at one of three label budgets, 2%, 5% or 20%, and the rest is the test set $T$. At 2% on COFFIN the combiners see 10 labeled positives.
- **Transductive.** Every method sees the predictions for all of $D_{\mathrm{ens}}$, test instances included, and labels for $L$ only. Metrics are computed on $T$.
- **Base classifiers.** Eleven heterogeneous, untuned scikit-learn models: Gaussian naive Bayes, linear discriminant analysis, logistic regression, a calibrated linear SVM, $k$-nearest neighbors, a decision tree, a random forest, extra trees, histogram gradient boosting, AdaBoost and a multilayer perceptron. Rounds 1 to 6 all use the set trained on base split 0.
- **The score.** PR-AUC (average precision) on $T$, the primary metric under heavy imbalance, with ROC-AUC reported beside it.
- **Where runs happen.** Every benchmark run goes to a RunPod GPU pod through [`ops/`](../../../../ops/README.md); the scripts are in [`examples/real_world/`](../../README.md).

---

## What comes next

Round 7 is planned as one pre-registered round on two freshly retrained sets of base classifiers (base splits 1 and 2), with the confidence type chosen on the labeled set by cross-validation. Its arms answer three questions at once, on identical splits:

- **Does round 4's method hold up** when the base classifiers are retrained and the confidence type is chosen without looking at test data?
- **Can the labels shape the factorization without the mismatch that sank the joint variants?** Rebuilt on round 4's ingredients, the candidates keep every representation the combiner is trained on label-blind, computed the way a test instance's is: a joint objective with a stop-gradient and a properly fitted combiner, the labels as one more row of the matrix, and training in episodes that hide a random part of the labeled set at each step.
- **How does CF compare with a pretrained tabular transformer?** TabPFN, given the transposed prediction matrix (one row per instance, one column per base classifier) with the labeled instances as context, is a training-free combiner. CF will be reported both transductively, as designed, and inductively, with new instances folded into a factorization fitted without them, so that the comparison with TabPFN, which does not use the test instances' predictions, is also made on equal terms.
