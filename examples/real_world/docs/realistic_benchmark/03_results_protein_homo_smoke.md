# Results: Smoke Test on `protein_homo`

**Run:** 2026-10-20/21 on a RunPod A40 · **Data:** KDD Cup 2004 protein homology, 145,751 rows,
0.889% positive · **Output:** `output/benchmarks/protein_homo/` (`results.csv`, `summary.csv`)

The goal was to validate the pipeline end to end. It also produced the first evidence on how
joint training behaves, and surfaced two bugs (see the end).

## Base classifiers (PR-AUC on D_ens, 72,876 rows, 648 positives)

| Model | PR-AUC | | Model | PR-AUC |
|---|---|---|---|---|
| extra_trees | 0.888 | | lda | 0.834 |
| hist_gbm | 0.886 | | knn | 0.837 |
| linear_svm | 0.879 | | decision_tree | 0.770 |
| random_forest | 0.876 | | gaussian_nb | 0.660 |
| mlp | 0.872 | | adaboost | 0.857 |
| logreg | 0.871 | | | |

Total base-model training: about 2 minutes on 9 vCPUs.

## Ensemble methods (PR-AUC on T, mean over 3 seeds)

Each CF row is one **fixed** configuration, not the best one picked on the test set.

| Method | 2% labels (13 pos) | 5% (32 pos) | 20% (130 pos) | Mean |
|---|---|---|---|---|
| ens_select (Caruana) | 0.8878 | 0.8917 | **0.8926** | **0.8907** |
| two_stage, d=4, uniform | **0.8881** | **0.8926** | 0.8880 | 0.8896 |
| stacking | 0.8815 | 0.8917 | 0.8901 | 0.8877 |
| two_stage, d=4, certainty | 0.8827 | 0.8907 | 0.8882 | 0.8872 |
| summa (no labels) | 0.8856 | 0.8862 | 0.8799 | 0.8839 |
| best_single | 0.8818 | 0.8871 | 0.8820 | 0.8836 |
| mean (no labels) | 0.8843 | 0.8848 | 0.8794 | 0.8828 |
| joint, d=4, uniform, ρ=0.1 | 0.8841 | 0.8846 | 0.8792 | 0.8826 |
| joint_sg, d=8, uniform, ρ=0.5 | 0.8814 | 0.8820 | 0.8769 | 0.8801 |
| joint_sum (original loss), d=2, certainty, ρ=0.5 | 0.8614 | 0.8642 | 0.8686 | 0.8647 |

Seed-to-seed standard deviation is 0.002–0.017 (largest for stacking at 2%), so **no method is
significantly better than another**. The whole field spans 0.876–0.893: this dataset is too easy
to separate the methods. It served its purpose as a smoke test; COFFIN is needed for a real verdict.

## Findings

1. **Two-stage is competitive and tends to help at the smallest budget.** With d=4 and uniform
   confidence it is +0.007 over stacking at 2% labels and ties at 5–20%. That is the
   variance-reduction effect the hypothesis predicts, but within seed noise.
2. **Uniform confidence beats certainty confidence** for two-stage at every d (+0.002 to +0.011),
   as predicted: certainty weights $\vert r - 0.5 \vert$ emphasize the easy negatives.
3. **Joint training currently collapses to the plain average.** Normalized joint runs score
   0.8843 / 0.8846 / 0.8792 against `mean` at 0.8843 / 0.8848 / 0.8794.
4. **The cause is labeled-point distortion, confirmed directly** (diagnostic at 5%, d=4, uniform):

   | ρ | Supervised loss (start → end) | Labeled-column shift | Test-column shift |
   |---|---|---|---|
   | 0.1 | 0.693 → 0.0002 | 0.205 | 0.003 |
   | 0.5 | 0.693 → 0.0006 | 0.071 | 0.002 |
   | 0.9 | 0.693 → 0.0021 | 0.037 | 0.002 |

   The labeled instance factors absorb the supervision (training loss near zero; labeled columns
   move 17–60× more than test columns), so the combiner $g$ learns little that transfers. At
   ρ=0.1 its test predictions rank exactly like the plain average (Spearman 0.9999).
5. **Stop-gradient removes the distortion but does not yet help.** Labeled and test columns then
   shift equally, the supervised loss settles at an honest 0.21, and the weights of $g$ move
   substantially, yet test PR-AUC stays at 0.883–0.886. The likely second bottleneck: the joint
   $g$ is an *unregularized* logistic regression fitted by Adam, whereas `stacking` and
   `two_stage` use scikit-learn's L2-regularized one.
6. **The original summed reconstruction loss is worst** (0.865): at this n it drowns the
   supervision. Why it trails two-stage is not yet tested; a likely reason is that its $g$, unlike
   the two-stage combiner, has no L2 regularization.
7. **Caruana ensemble selection is the strongest baseline**, as expected from the literature.

## Scoring the pre-registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Tree ensembles, linear models and MLP lead; Gaussian NB worst | ✅ |
| 2 | two_stage ≈ stacking with d ≥ m | ✅ (Δ 0.001 at d=20 in the first run) |
| 3 | joint_sum ≈ two_stage | ❌ worse than two_stage at d < m (likely: no L2 on $g$; untested) |
| 4 | joint ρ=0.1 worst; ρ=0.9 ≈ two_stage | ◐ the mechanism is confirmed, but *every* ρ collapses to the mean |
| 5 | mean / best_single ≥ stacking at 2% | ◐ true on average (0.884 / 0.882 vs 0.882), not in every seed |
| 6 | No significant CF win | ✅ |
| — | Certainty confidence counterproductive | ✅ uniform wins |

## Bugs found and fixed

- **NaN in the joint trainer.** The supervised loss clamped probabilities at $1 - 10^{-15}$, which
  is exactly 1.0 in float32, so saturated predictions gave $\log 0$ and NaN weights (around epoch
  810 at ρ=0.9). It now uses the numerically stable logit form of binary cross-entropy.
- **Ops wrapper hid a live pod.** When a job failed after provisioning, `launch()` reported
  "NOT provisioned", leaving an A40 billing. It now asks SkyPilot whether the cluster exists.

## Next steps

1. Add L2 regularization on $g$'s weights in the joint trainer (matching the scikit-learn combiner),
   then rerun joint and joint_sg. Stop-gradient plus a regularized $g$ is the cleanest test of
   whether label-steered factorization helps.
2. Move to `coffin_splice`, where base models are weak (PR-AUC around 0.3–0.5) and there is room
   for the methods to separate. Use more seeds (5+) at the smallest budget.
3. Report CF methods with hyperparameters chosen on L (nested), never on T.
