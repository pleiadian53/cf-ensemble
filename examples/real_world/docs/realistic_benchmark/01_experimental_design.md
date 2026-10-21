# Experimental Design

## Datasets

| Dataset | Rows | Positives | Features | Role |
|---|---|---|---|---|
| `protein_homo` — KDD Cup 2004 protein homology (imbalanced-learn mirror) | 145,751 | ~0.9% | 74 tabular | Smoke test of the full pipeline |
| `coffin_splice` — human acceptor splice sites (Sonnenburg & Franc 2010, "COFFIN") | 1M subset (of 50M) | to be measured (~0.3–1.5% reported) | 141-nt DNA window → one-hot + 3-mer spectra | Headline benchmark |
| LargeGOPred (Wang et al. 2018), planned | 63k proteins × 277 GO terms | ~0.3–2% per term | 8,000 3-mer frequencies | Second benchmark, many tasks |

COFFIN continues the lineage of the original CF-ensemble experiments, which used splice-site
data from the same group (Rätsch et al. 2007; Schweikert et al. 2009); those exact files are
no longer hosted. The archive (1.7 GB, md5-verified) is streamed so only members up to 500 MB
are extracted; the 10M/50M-example files are skipped.

## Splits: two levels, no leakage

```
dataset ──stratified──▶ D_base (50%)  → trains the base classifiers
                     └▶ D_ens  (50%)  → R[u, i] = P(y=1) from classifier u on instance i
                                          └─stratified per seed─▶ L (labeled, budget %) | T (test)
```

- **D_base and D_ens are disjoint**, so R carries honest out-of-sample errors (the noise
  CF-Ensemble is meant to remove) and no label crosses levels.
- **Transductive:** every method sees R for all of D_ens, but labels only for L. Metrics are on T.
- **Label budgets:** 2%, 5%, 20% of D_ens. At ~0.9% positives, 2% is about 13 labeled positives,
  the few-label regime where the reconstruction term is supposed to help most.

## Base classifiers

Eleven deliberately heterogeneous, untuned models (`base_models.py`): Gaussian NB, LDA,
logistic regression, calibrated linear SVM, kNN, decision tree, random forest, extra trees,
histogram gradient boosting, AdaBoost, MLP. They mirror the 12 diverse WEKA classifiers of the
original experiments: complementary, miscalibrated "teachers".

## Methods

All supervised combiners use logistic regression as $g$, class-balanced like the CF
trainers' supervised loss.

| Method | Uses labels | What it does |
|---|---|---|
| `mean` | no | Average of the m base predictions |
| `best_single` | L, for selection | Base model with the best PR-AUC on L |
| `stacking` | L | Logistic regression on raw columns $r_{\cdot i}$ |
| `two_stage` | L, after factorizing | Factorize with $\rho = 1$ (labels unused), then logistic regression on $\hat{r}_ {\cdot i}$ |
| `joint_rho<ρ>` | L, during factorizing | PyTorch CF-Ensemble: $\rho L_{\text{recon}} + (1-\rho) L_{\text{sup}}$, with $g$ learned jointly |
| `joint_sum_rho0.5` | L | Same, with the original summed reconstruction loss (diagnostic) |

**`two_stage` and `joint` share the same factorization model and optimizer** (PyTorch,
Adam, full batch). Their gap therefore isolates one thing: whether labels steer the
factorization.

## Design decisions

### Normalizing the reconstruction loss

The PyTorch trainer *sums* the reconstruction loss over all $m \cdot n$ entries but *averages*
the supervised loss over labeled points. With $m = 11$ and $n \approx 73{,}000$, reconstruction
outweighs supervision by roughly $10^5$ before $\rho$ is applied, so joint training would
silently reduce to two-stage. The trainer now has an opt-in `normalize_recon=True`, which
divides reconstruction and regularization by $m \cdot n$, so $\rho$ balances two
per-entry quantities. The default is unchanged so earlier results still reproduce; the
`joint_sum_rho0.5` variant keeps the old behavior as a diagnostic.

### Latent dimension must be below m

$\hat{R} = X^\top Y$ has rank at most $\min(d, m)$. With $d \ge m$ the factorization can
reproduce $R$ exactly: there is no low-rank denoising, only shrinkage from $\lambda$. The smoke
test uses $d = 20 \gt m = 11$ and is only a pipeline check; real runs sweep $d \in \lbrace 2, 4, 8 \rbrace$.

### Few observations per instance

Each classifier factor $x_u$ is fitted from $n$ observations, but each instance factor $y_i$
from only $m \approx 11$. The low-rank bottleneck is what makes the model denoise, so
anything that adds flexibility per instance (larger $d$, more expressive interaction
functions) needs a matching constraint. See [02_hypotheses_and_predictions.md](02_hypotheses_and_predictions.md).

## Metrics

PR-AUC (average precision) is primary under heavy imbalance; ROC-AUC is reported alongside.
Each (budget, seed) cell draws a fresh stratified L/T split; results report mean ± std
over seeds.

## Planned extensions

- Confidence ablation: certainty $\vert r - 0.5 \vert$ vs. uniform weights
- Stop-gradient variant: supervision updates $X$ and $g$, not the labeled $y_i$
- Baselines: Caruana greedy ensemble selection; Spectral Meta-Learner (label-free)
- Logistic link with biases, $\hat{r}_ {ui} = \sigma(x_u^\top y_i + b_u + b_i)$, fitted with cross-entropy
- Amortized instance factors $y_i = f(\text{features}_ i)$
