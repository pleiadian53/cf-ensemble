# Results: Pre-Registered Confirmation

**Run:** 2026-10-21 on a RunPod A40 · **Pre-registration:** [06](06_confirmation_preregistration.md)
(committed `bcdfb0c` before the run) · **Output:** `output/benchmarks/<dataset>/confidence_link_base0_confirm/`
· Fresh seeds 5–14 × budgets 2/5/20% = 30 paired cells per dataset

## Primary hypotheses

| | Dataset | Configuration | Mean Δ vs. raw `ens_select` | Wins | One-sided Wilcoxon p | Holm-adjusted | Verdict |
|---|---|---|---|---|---|---|---|
| **H1** | protein_homo | `cf_ens_select` · learned · logit · d=4 | **+0.0033** | 19/30 | 0.0098 | 0.0098 | **Confirmed** |
| **H2** | coffin_splice | `cf_ens_select` · classifier · logit · d=8 | **+0.0046** | 23/30 | 0.0073 | 0.0145 | **Confirmed** |

Both configurations beat the strongest baseline on seeds that played no part in choosing them.

Per budget (mean Δ):

| | 2% | 5% | 20% |
|---|---|---|---|
| H1 (protein_homo) | +0.0025 | +0.0028 | +0.0045 |
| H2 (coffin_splice) | +0.0080 | +0.0034 | +0.0023 |

On COFFIN, H2's raw PR-AUC is 0.286 vs. 0.282 for ensemble selection, 0.271 for stacking, and 0.262 for
the plain average.

## Cross-dataset results (secondary)

| Configuration | protein_homo Δ | coffin_splice Δ |
|---|---|---|
| learned · logit · d=4 (H1) | **+0.0033** (p = 0.010) | −0.0249 (4/30 wins) |
| classifier · logit · d=8 (H2) | +0.0009 (p = 0.06, 21/30) | **+0.0046** (p = 0.007) |

**The best confidence type depends on the data.** Learned cell-level confidence needs enough labeled
positives and reasonably strong base models (protein_homo); with weak base models and 10–104 positives
(COFFIN) it hurts badly. Per-classifier confidence is the safer default: it is the only configuration
at or above raw ensemble selection on both datasets.

## Exploratory: shrinkage and combined confidence

| Configuration | protein_homo Δ (wins, p) | coffin_splice Δ (wins, p) |
|---|---|---|
| classifier_shrunk · logit · d=4 | +0.0032 (23/30, p = 0.0009) | −0.0059 (9/30) |
| classifier_shrunk · logit · d=8 | +0.0015 (24/30, p = 0.010) | −0.0010 (21/30, p = 0.06) |
| combined · logit · d=4 / d=8 | +0.0017 / +0.0012 | −0.0205 / −0.0187 |

- My prediction that shrinkage would rescue the 2% budget on COFFIN was **wrong**: unshrunk classifier
  confidence already does well there on these seeds (+0.008), and shrinking it costs about 0.010.
- Shrinkage looks promising on protein_homo (d=4: 23/30 wins), but that is exploratory and would need its
  own confirmation.
- Multiplying in the learned cell reliability (`combined`) inherits its weakness on COFFIN.

## Scoring the pre-registered predictions

| Prediction | Outcome |
|---|---|
| H1 confirmed (I said ~45%) | ✅ confirmed |
| H2 confirmed (I said ~20%) | ✅ confirmed: I underestimated it |
| Shrunk classifier confidence beats unshrunk at 2% on COFFIN | ❌ the reverse |

## What this establishes, and what it does not

**Established:** a CF representation (logistic-link factorization with classifier and instance biases,
two-stage, with labels used only to weight entries through $C$) followed by ensemble selection on
$\hat{R}$ beats ensemble selection on the raw predictions, on held-out seeds, on two datasets, with the
right confidence type.

**Not yet established:**

- **Size:** the gains are small (about +0.003 to +0.005 PR-AUC; 1–2% relative).
- **Robustness to the base layer:** every run reuses one set of base classifiers (`base_seed0`). The next
  confirmation should retrain them on new D_base / D_ens splits.
- **Novelty of the combiner:** ensemble selection is applied after the fact. Building the selection
  bias into CF itself is the subject of `04_joint_simplex.py` (results doc 08).
- **Choosing the confidence type without peeking:** it is picked per dataset here. A principled rule
  (for example, by the number of labeled positives or base-model strength), or choosing it on L by
  cross-validation, is needed before this is a usable method.
