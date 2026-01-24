# CF-Ensemble Optimization: Knowledge Distillation Meets Collaborative Filtering

**From soft targets to matrix factorization: A unified framework for ensemble learning**

---

## Introduction

In the [knowledge distillation tutorial](knowledge_distillation_tutorial.md), we learned that effective learning combines two objectives:
1. **Imitation**: Match soft targets from a teacher
2. **Supervision**: Match hard labels from ground truth

This tutorial reveals a surprising connection: **ensemble learning through collaborative filtering follows the exact same principle**. Instead of distilling a single teacher model, we distill knowledge from an *ensemble* of base models through matrix factorization.

The key insight is that the probability matrix—where base classifiers act as "teachers" for different data points—can be decomposed to reveal latent factors that capture both:
- **Reconstruction fidelity**: Faithful representation of ensemble predictions
- **Predictive accuracy**: Alignment with true labels

This document develops the mathematical framework for CF-based ensemble learning, showing how knowledge distillation principles generalize to heterogeneous ensembles.

---

## 1. From Neural Networks to Ensembles: The Structural Analogy

### Knowledge Distillation Recap

In KD, we have:
- **Teacher**: Large model with soft predictions $q_t$
- **Student**: Small model learning from $q_t$ and hard labels $y_g$
- **Loss**: $\mathcal{L}_{\text{KD}} = \rho \cdot L(\text{soft}) + (1-\rho) \cdot L(\text{hard})$

### CF-Ensemble Parallel

In CF-ensemble, we have:
- **"Teachers"**: Base classifier predictions forming probability matrix $R$
- **"Student"**: Latent factor model reconstructing $R$ and predicting labels
- **Loss**: $\mathcal{L}_{\text{CF}} = \rho \cdot L(\text{recon}) + (1-\rho) \cdot L(\text{supervised})$

**The skeleton is identical—only the organs differ.**

| Concept | Knowledge Distillation | CF-Ensemble |
|---------|------------------------|-------------|
| Teacher knowledge | Soft probability distribution | Probability matrix $R$ |
| Student model | Small neural network | Latent factors $X, Y$ |
| Soft matching | KL divergence | Reconstruction loss |
| Hard matching | Cross-entropy with labels | Supervised aggregation |
| Trade-off | $\rho$ | $\rho$ |

---

## 2. The Collaborative Filtering View of Ensemble Learning

### Borrowing from Recommender Systems

In collaborative filtering for recommender systems:
- **Users** rate **items**
- We factorize the rating matrix to find latent preferences
- Predictions are reconstructed via dot products of latent vectors

In our ensemble context:
- **Base classifiers** (users) "rate" **data points** (items)
- "Ratings" are predicted probabilities $P(y=1 \mid x)$
- We factorize to find latent factors explaining predictions

This is more than analogy—it's a direct mathematical mapping.

### The Probability Matrix

Define the **probability (rating) matrix**:
$$R \in [0,1]^{m \times n}$$

where:
- $m$ = number of base classifiers
- $n$ = number of data points (train + test)
- $r_{ui}$ = classifier $u$'s predicted probability for point $i$

$$r_{ui} = P_u(y=1 \mid x_i)$$

This matrix encodes the entire ensemble's predictions across all data points.

---

## 3. Matrix Factorization: Finding Latent Structure

### Why Factorize?

The probability matrix $R$ is noisy and redundant:
- Different classifiers make correlated errors
- Some patterns are genuine signal (related to true labels)
- Other patterns are noise (systematic biases, overfitting)

Matrix factorization **separates signal from noise** by finding low-dimensional latent representations.

### The Factorization Model

We approximate $R$ using latent vectors:

**Classifier latent factors**:
$$X = [x_1, x_2, \ldots, x_m] \in \mathbb{R}^{d \times m}$$

where $x_u \in \mathbb{R}^d$ is the $d$-dimensional latent vector for classifier $u$.

**Instance latent factors**:
$$Y = [y_1, y_2, \ldots, y_n] \in \mathbb{R}^{d \times n}$$

where $y_i \in \mathbb{R}^d$ is the $d$-dimensional latent vector for data point $i$.

### Reconstruction via Inner Product

The predicted probability is:
$$\hat{r}_{ui} = x_u^\top y_i = \sum_{k=1}^d x_{uk} \cdot y_{ik}$$

This dot product measures **alignment in latent space**: classifiers and data points with similar latent factors produce similar probabilities.

---

## 4. The Reconstruction Loss: Matching the Ensemble

### Basic Squared Error

The simplest reconstruction objective is weighted squared error:

$$L_{\text{recon}}(X, Y) = \sum_{u=1}^m \sum_{i=1}^n c_{ui} \left( r_{ui} - x_u^\top y_i \right)^2$$

where $c_{ui} > 0$ are **confidence weights** (discussed below).

**Interpretation**: Find latent factors that faithfully reproduce the observed probabilities.

### Regularization

To prevent overfitting, we add $\ell_2$ regularization:

$$L_{\text{recon}}(X, Y) = \sum_{u=1}^m \sum_{i=1}^n c_{ui} \left( r_{ui} - x_u^\top y_i \right)^2 + \lambda \left( \sum_{u=1}^m \|x_u\|^2 + \sum_{i=1}^n \|y_i\|^2 \right)$$

where $\lambda > 0$ controls regularization strength.

**This is analogous to the teacher-matching term in KD**: we're learning to imitate the ensemble's predictions.

---

## 5. Confidence Weights: Not All Predictions Are Equal

### The Role of $C$

The confidence matrix $C \in \mathbb{R}_+^{m \times n}$ encodes our trust in each probability:

$$c_{ui} = \text{confidence}(r_{ui})$$

Higher $c_{ui}$ means:
- We trust classifier $u$'s prediction for point $i$ more
- Reconstruction error on this entry is weighted more heavily
- Latent factors are pulled to match this prediction more strongly

### How to Define Confidence

Several approaches work:

#### 1. Calibration-Based Confidence
Use calibration metrics like **Brier score**:
$$c_{ui} = 1 - \text{Brier}_u = 1 - \frac{1}{N}\sum_{j} (r_{uj} - y_j)^2$$

Better-calibrated classifiers get higher weight.

#### 2. Prediction Certainty
Use distance from 0.5 (uncertainty):
$$c_{ui} = |r_{ui} - 0.5|$$

Confident predictions (close to 0 or 1) get higher weight.

#### 3. Ensemble Agreement
Use variance across classifiers:
$$c_{ui} = 1 - \text{Var}_u(r_{\cdot i})$$

Points with high ensemble agreement get higher weight.

#### 4. Label-Aware Confidence (for labeled data)
For $i \in \mathcal{L}$ (labeled points):
$$c_{ui} = \begin{cases}
r_{ui} & \text{if } y_i = 1 \text{ (reward correct high predictions)} \\
1 - r_{ui} & \text{if } y_i = 0 \text{ (reward correct low predictions)}
\end{cases}$$

This explicitly upweights predictions consistent with labels.

### Critical Insight

**The confidence matrix is how we encode which predictions are "signal" vs "noise"**. Without it, reconstruction treats all probabilities equally, including systematic errors we want to suppress.

---

## 6. Train-Test Split and Transductive Learning

### Data Partitioning

We split data points into:
- **Labeled set** $\mathcal{L}$: Training data with known labels $y_i$
- **Unlabeled set** $\mathcal{U}$: Test data with masked labels

Crucially, **both sets are used during training**, but labels are only available for $\mathcal{L}$.

This is **transductive** or **semi-supervised** learning: we observe test inputs (and their base model predictions) at training time, but not their labels.

### Why This Makes Sense for Ensemble Learning

Unlike typical ML, in ensemble contexts:
- Base models have *already* made predictions on test data
- We have access to the probability matrix $R$ for all points
- We want to learn how to aggregate these existing predictions

The transductive setting is **natural for this problem**: we're not training base models, we're learning to combine their outputs.

### Constraints

During training:
- **Classifier factors $X$**: Learned from all data (train + test)
- **Train instance factors $Y_{\mathcal{L}}$**: Learned using reconstruction + supervision
- **Test instance factors $Y_{\mathcal{U}}$**: Learned using reconstruction only (no labels)

---

## 7. The Supervised Loss: Learning What "Signal" Means

### The Aggregation Function

For each data point $i$, we aggregate reconstructed probabilities into a final prediction.

Collect the reconstructed probabilities across all classifiers:
$$\hat{r}_{\cdot i} = [\hat{r}_{1i}, \hat{r}_{2i}, \ldots, \hat{r}_{mi}] \in \mathbb{R}^m$$

Define an **aggregation function** $g: \mathbb{R}^m \to [0,1]$:
$$\hat{p}_i = g(\hat{r}_{\cdot i})$$

This is our final predicted probability for point $i$.

### Aggregation Choices

**Simple mean**:
$$g(\hat{r}_{\cdot i}) = \frac{1}{m} \sum_{u=1}^m \hat{r}_{ui}$$

**Weighted mean** (learnable weights $w$):
$$g(\hat{r}_{\cdot i}) = \sigma\left( w^\top \hat{r}_{\cdot i} + b \right)$$
where $\sigma$ is sigmoid.

**Stacker model** (meta-learner):
$$g(\hat{r}_{\cdot i}) = \text{MLP}(\hat{r}_{\cdot i})$$

**Note**: Simpler is often better to avoid overfitting. Start with mean or weighted mean.

### The Supervised Loss

For labeled points, we use binary cross-entropy:

$$L_{\text{sup}}(X, Y, \theta) = \sum_{i \in \mathcal{L}} \text{CE}(y_i, g_\theta(\hat{r}_{\cdot i}))$$

where:
$$\text{CE}(y_i, \hat{p}_i) = -y_i \log \hat{p}_i - (1-y_i) \log(1-\hat{p}_i)$$

and $\theta$ represents parameters of the aggregator $g_\theta$.

**This is analogous to the hard label term in KD**: we're ensuring the model predicts the correct labels, not just reconstructs probabilities.

---

## 8. The Complete Objective: Putting It All Together

### The KD-Style Combined Loss

$$\boxed{
\mathcal{L}_{\text{CF-Ensemble}}(X, Y, \theta) = \rho \cdot L_{\text{recon}}(X, Y) + (1-\rho) \cdot L_{\text{sup}}(X, Y, \theta)
}$$

Expanding the components:

$$\begin{align}
\mathcal{L}_{\text{CF-Ensemble}} &= \rho \left[ \sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2) \right] \\
&\quad + (1-\rho) \sum_{i \in \mathcal{L}} \text{CE}(y_i, g_\theta(\hat{r}_{\cdot i}))
\end{align}$$

where:
- **First term**: Matrix reconstruction (all points, weighted by confidence)
- **Second term**: Supervised prediction (labeled points only)
- **$\rho \in [0,1]$**: Trade-off between reconstruction fidelity and predictive accuracy

### Interpretation of $\rho$

- **$\rho = 1$**: Pure matrix factorization (no supervision)
  - Faithfully reconstructs probabilities
  - No guarantee of good predictions
  - Reproduces base model mistakes

- **$\rho = 0$**: Pure supervised stacking (no CF)
  - Learns aggregation from labels only
  - Ignores probability structure
  - Reduces to standard stacking

- **$\rho \in (0.3, 0.7)$**: Balanced approach
  - Leverages both probability structure and labels
  - Learns which patterns are signal vs noise
  - **Recommended starting range**

---

## 9. Why This Formulation Addresses Previous Failures

### The Problem with Pure Reconstruction

Your earlier attempts used primarily $L_{\text{recon}}$, which has a fundamental flaw:

> **Optimizing squared error on probabilities encourages reconstructing the ensemble *as is*, including systematic errors.**

If multiple base models consistently misclassify certain regions:
- The reconstruction will faithfully reproduce these errors
- Low-rank factorization will *smooth* these errors across similar points
- The result: amplification of systematic biases

### How Supervision Fixes This

Adding $L_{\text{sup}}$ teaches the system **what "signal" means**:
- Reconstruction patterns consistent with labels → amplified
- Reconstruction patterns inconsistent with labels → suppressed
- The model learns to distinguish true predictive signal from noise

This is exactly why KD combines soft and hard targets!

### The Role of Confidence Weights

For labeled data ($i \in \mathcal{L}$), using label-aware confidence:
$$c_{ui} = \begin{cases}
r_{ui} & \text{if } y_i = 1 \\
1 - r_{ui} & \text{if } y_i = 0
\end{cases}$$

means reconstruction preferentially matches **correct predictions**:
- True Positives (high $r_{ui}$ for $y_i=1$) → high weight
- True Negatives (low $r_{ui}$ for $y_i=0$) → high weight
- False Positives (high $r_{ui}$ for $y_i=0$) → low weight
- False Negatives (low $r_{ui}$ for $y_i=1$) → low weight

This is your "TP/TN amplification, FP/FN suppression" idea, formalized.

---

## 10. Optimization: Alternating Least Squares (ALS)

### The ALS Framework

With reconstruction term in quadratic form, we can derive closed-form updates by alternating:

**Fix $Y$, update $X$**:
For each classifier $u$:
$$x_u = \left( Y C_u Y^\top + \lambda I \right)^{-1} Y C_u r_u$$

where:
- $C_u = \text{diag}(c_{u1}, \ldots, c_{un})$ (confidences for classifier $u$)
- $r_u = [r_{u1}, \ldots, r_{un}]^\top$ (probabilities from classifier $u$)

**Fix $X$, update $Y$**:
For each point $i$:
$$y_i = \left( X C_i X^\top + \lambda I \right)^{-1} X C_i r_i$$

where:
- $C_i = \text{diag}(c_{1i}, \ldots, c_{mi})$ (confidences for point $i$)
- $r_i = [r_{1i}, \ldots, r_{mi}]^\top$ (probabilities for point $i$)

### Handling the Supervised Term

After each ALS step, update the aggregator $\theta$ via gradient descent on $L_{\text{sup}}$:
$$\theta \leftarrow \theta - \eta \nabla_\theta L_{\text{sup}}(X, Y, \theta)$$

**Algorithm summary**:
```
1. Initialize X, Y randomly
2. For each epoch:
   a. Fix Y, update X via ALS (all classifiers)
   b. Fix X, update Y via ALS (all points)
   c. Fix X, Y, update θ via gradient descent (labeled points only)
3. Return X, Y, θ
```

### Computational Efficiency

- Each ALS update: $O(d^3 + d^2n)$ or $O(d^3 + d^2m)$
- Parallelizable across classifiers / data points
- Typically converges in 10-50 iterations
- Much faster than gradient descent on full objective

---

## 11. Test-Time Inference

### Reusing Classifier Factors

At test time:
- **Classifier factors $X$ are fixed** (classifiers don't change)
- **Test point factors $Y_{\mathcal{U}}$ are already learned** (from transductive training)
- Simply aggregate: $\hat{p}_i = g_\theta(\hat{r}_{\cdot i})$ where $\hat{r}_{ui} = x_u^\top y_i$

### Truly New Points (Inductive Setting)

For a completely new point $i_{\text{new}}$ not seen during training:

1. Obtain base model predictions: $r_{1,i_{\text{new}}}, \ldots, r_{m,i_{\text{new}}}$
2. Solve for its latent factor (fix $X$):
   $$y_{i_{\text{new}}} = (X C_{i_{\text{new}}} X^\top + \lambda I)^{-1} X C_{i_{\text{new}}} r_{i_{\text{new}}}$$
3. Reconstruct: $\hat{r}_{u,i_{\text{new}}} = x_u^\top y_{i_{\text{new}}}$
4. Aggregate: $\hat{p}_{i_{\text{new}}} = g_\theta(\hat{r}_{\cdot, i_{\text{new}}})$

This is analogous to "cold-start" solutions in recommender systems.

---

## 12. Connection to Other Ensemble Methods

### vs. Simple Averaging

**Simple averaging**: $\hat{p}_i = \frac{1}{m}\sum_u r_{ui}$
- Treats all classifiers equally
- No adaptation to data regions
- Our method: learns **context-dependent weights** via latent factors

### vs. Weighted Averaging

**Weighted averaging**: $\hat{p}_i = \sum_u w_u r_{ui}$
- Global weights for each classifier
- No adaptation to specific points
- Our method: weights are **instance-specific** via $x_u^\top y_i$

### vs. Stacking

**Stacking**: Train meta-learner $g(r_{1i}, \ldots, r_{mi}) \to y_i$
- Can overfit if not careful
- Doesn't leverage unlabeled data structure
- Our method: **regularizes via reconstruction** and uses transductive information

### vs. Boosting

**Boosting**: Sequential training with re-weighting
- Requires training base models sequentially
- Not applicable to pre-trained heterogeneous ensembles
- Our method: works with **any pre-trained base models**

---

## 13. Theoretical Intuition: Why Low-Rank Helps

### The Low-Rank Prior

By using $d \ll \min(m, n)$ latent dimensions, we enforce:

$$\text{rank}(\hat{R}) \leq d$$

This **low-rank constraint acts as regularization**:
- Separates signal (low-rank patterns) from noise (high-rank residuals)
- Encourages generalization to similar points
- Learns shared structure across classifiers

### What the Latent Factors Capture

**Classifier factors $x_u$** encode:
- Which types of patterns classifier $u$ is good at
- Its systematic biases (overfitting tendencies)
- Its complementarity with other classifiers

**Instance factors $y_i$** encode:
- What "type" of point $i$ is (easy, hard, ambiguous)
- Which classifiers are likely reliable for this point
- Its position in the difficulty landscape

The dot product $x_u^\top y_i$ measures **compatibility**: how reliable is classifier $u$ for point $i$?

---

## 14. Advanced Variants and Extensions

### Alternative Reconstruction Losses

Instead of squared error, use **Bernoulli log-likelihood** (since $r_{ui} \in [0,1]$):

$$L_{\text{recon}} = -\sum_{u,i} c_{ui} \left[ r_{ui} \log \hat{r}_{ui} + (1-r_{ui}) \log(1-\hat{r}_{ui}) \right]$$

This often produces **sharper** distinctions (less averaging).

### Incorporating Additional Features

Extend latent factors with side information:
$$\hat{r}_{ui} = x_u^\top y_i + a_u^\top f_i + b_i$$

where $f_i$ are features of point $i$ (e.g., input features, metadata).

### Hierarchical Structures

Group similar classifiers and learn group-level factors:
$$x_u = \beta_u x_{\text{group}(u)} + \epsilon_u$$

Encourages parameter sharing and handles large ensembles better.

### Attention Mechanisms

Replace dot product with learned attention:
$$\hat{r}_{ui} = \text{Attention}(x_u, y_i) = \text{softmax}(x_u^\top W y_i)$$

Allows more flexible interactions.

---

## 15. Practical Implementation Guide

### Hyperparameters

1. **Latent dimension $d$**: Start with $d \in [10, 50]$
   - Too small: underfitting
   - Too large: overfitting
   - Rule of thumb: $d \approx \sqrt{m}$ or cross-validate

2. **Regularization $\lambda$**: Start with $\lambda \in [0.01, 0.1]$
   - Adjust based on training set size
   - Higher $\lambda$ for smaller datasets

3. **Trade-off $\rho$**: Start with $\rho = 0.5$
   - Increase if base models are reliable
   - Decrease if labels are noisy

4. **Temperature** (if using Bernoulli loss): Start with $T = 1$
   - Similar to KD, can soften distributions

### Validation Strategy

Use cross-validation on the labeled set $\mathcal{L}$:
- Split $\mathcal{L}$ into train/validation
- Train on $\mathcal{L}_{\text{train}} \cup \mathcal{U}$ (transductive)
- Validate on $\mathcal{L}_{\text{val}}$
- Select hyperparameters maximizing validation performance

### Computational Considerations

**Memory**: Store $X \in \mathbb{R}^{d \times m}$, $Y \in \mathbb{R}^{d \times n}$
- Total: $O(d(m+n))$ space
- Much smaller than full $R$ if $d \ll \min(m,n)$

**Time per iteration**:
- ALS update: $O(d^2(m+n) + d^3(m+n))$
- Aggregator update: $O(|\mathcal{L}| \cdot m)$
- Typical: seconds to minutes for moderate-sized problems

---

## 16. Diagnostic Tools and Debugging

### Check Reconstruction Quality

Monitor reconstruction error separately on train/test:
$$\text{RMSE}_{\mathcal{L}} = \sqrt{\frac{1}{m|\mathcal{L}|}\sum_{u,i \in \mathcal{L}} (r_{ui} - \hat{r}_{ui})^2}$$

$$\text{RMSE}_{\mathcal{U}} = \sqrt{\frac{1}{m|\mathcal{U}|}\sum_{u,i \in \mathcal{U}} (r_{ui} - \hat{r}_{ui})^2}$$

Large gap suggests overfitting to labeled structure.

### Visualize Latent Spaces

- **t-SNE/UMAP of $Y$**: Do labeled points cluster by class?
- **Classifier similarities**: $\text{sim}(x_u, x_v) = \frac{x_u^\top x_v}{\|x_u\|\|x_v\|}$
- **Point difficulties**: $\|y_i\|$ (higher norm → more unusual)

### Analyze Learned Weights

For weighted aggregation, examine learned $w$:
- Which classifiers get highest weight?
- Does this match expected reliability?
- Are weights interpretable?

---

## 17. When Will This Work vs. Stacking?

### CF-Ensemble Advantages

**Works better when**:
- Base models have **complementary errors** (different failure modes)
- Large unlabeled test set with structure (benefits from transduction)
- Base models are **diverse** (different architectures, features)
- Moderate labeled data (regularization via reconstruction helps)

**Example**: Medical diagnosis with multiple modalities (imaging, labs, clinical notes) where each model excels in different patient subgroups.

### Stacking Advantages

**Works better when**:
- Very large labeled training set
- Base models are highly reliable and well-calibrated
- Test distribution differs significantly from train (distribution shift)
- Need strict inductive guarantees

**Example**: Standard benchmark tasks with abundant labeled data.

### Best of Both Worlds

Hybrid approach:
1. Use CF-ensemble for transductive test set
2. Train stacker as backup for out-of-distribution points
3. Detect distribution shift and route accordingly

---

## 18. Mathematical Summary

### Complete Formulation

**Given**:
- Probability matrix $R \in [0,1]^{m \times n}$
- Confidence matrix $C \in \mathbb{R}_+^{m \times n}$
- Labels $\{y_i\}_{i \in \mathcal{L}}$ for labeled set $\mathcal{L} \subset \{1,\ldots,n\}$

**Optimize**:
$$\min_{X \in \mathbb{R}^{d \times m}, Y \in \mathbb{R}^{d \times n}, \theta} \mathcal{L}_{\text{CF-Ensemble}}(X, Y, \theta)$$

where:
$$\begin{align}
\mathcal{L}_{\text{CF-Ensemble}} &= \rho \left[ \sum_{u=1}^m \sum_{i=1}^n c_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2) \right] \\
&\quad + (1-\rho) \sum_{i \in \mathcal{L}} \left[ -y_i \log g_\theta(\hat{r}_{\cdot i}) - (1-y_i) \log(1-g_\theta(\hat{r}_{\cdot i})) \right]
\end{align}$$

**Prediction**:
For point $i$ (train or test):
$$\hat{p}_i = g_\theta(\hat{r}_{\cdot i}) \quad \text{where} \quad \hat{r}_{ui} = x_u^\top y_i$$

---

## 19. Philosophical Perspective

### Learning from Behavior, Not Just Labels

Like knowledge distillation, CF-ensemble embodies a key principle:

> **Models learn more from *how* other models think than from *what* the correct answer is.**

The probability matrix $R$ encodes:
- Patterns of agreement and disagreement
- Regions of confidence and uncertainty
- Complementary expertise across models

By factorizing $R$ while supervising on labels, we:
- Discover latent structure in ensemble behavior
- Learn which patterns are signal vs noise
- Build predictions that **transcend individual model limitations**

### From Compression to Composition

KD compresses a single model; CF-ensemble **composes multiple models**:
- Each base model contributes partial knowledge
- Latent factors discover how to combine them
- The result can exceed any individual model

This is ensemble learning in its purest form: **the whole is greater than the sum of parts**.

---

## 20. Research Directions and Open Questions

### Theoretical Questions

1. **Generalization bounds**: How does transductive access affect generalization?
2. **Sample complexity**: How many labeled examples are needed for reliable latent factors?
3. **Identifiability**: Are learned factors unique (up to rotation)?

### Algorithmic Improvements

1. **Non-linear factorization**: Replace $x_u^\top y_i$ with neural networks
2. **Dynamic ensembles**: Update $X$ when new classifiers are added
3. **Active learning**: Which points should we label to maximally improve $Y$?

### Applications

1. **Deep ensembles**: Apply to neural network ensembles with thousands of models
2. **Multi-task learning**: Share latent factors across related tasks
3. **Federated learning**: Learn $X$ without sharing raw predictions

---

## 21. Conclusion

We've developed a unified framework connecting knowledge distillation and collaborative filtering for ensemble learning. The key insights are:

1. **Structural analogy**: KD's soft-hard combination maps directly to CF-ensemble's reconstruction-supervision combination

2. **Mathematical formulation**: The loss $\mathcal{L} = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$ balances matrix fidelity with predictive accuracy

3. **Practical advantages**: Leverages unlabeled test structure, learns instance-specific weights, and regularizes through low-rank factorization

4. **Why it should work**: Unlike pure reconstruction, adding supervision teaches the model to distinguish signal from noise

This framework addresses the fundamental limitation of previous CF-ensemble approaches: **faithfully reconstructing the probability matrix isn't enough—we must reconstruct it in a way that aligns with true labels**.

---

## References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
2. Hinton, G., et al. (2015). *Distilling the Knowledge in a Neural Network*. NIPS Workshop.
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.

---

**Next Steps**: Implement this framework and empirically test whether the KD-inspired combined objective finally makes CF-ensemble learning work! See the implementation guide in `notebooks/` and source code in `src/cfensemble/`.
