# Cell-Level Reliability: Polarity Models and Confidence Weighting

**From global reconstruction to fine-grained trust: Learning which classifier-point pairs to believe**

---

## Introduction

In the [CF-Ensemble optimization tutorial](cf_ensemble_optimization_objective_tutorial.md), we introduced confidence weights $c_{ui}$ that control how much we trust each classifier's prediction for each data point. But we left open the question: **How should we set these weights?**

This tutorial explores a sophisticated approach: **learning a polarity model** that predicts whether each cell $(u,i)$ in the probability matrix corresponds to a correct or incorrect prediction. This provides:
1. **Massive supervision**: $m \times |\mathcal{L}|$ cell-level training examples instead of just $|\mathcal{L}|$ point-level examples
2. **Fine-grained trust**: Direct modeling of "which classifier to trust where"
3. **Explicit signal-noise separation**: Goes beyond hoping low-rank reconstruction magically filters errors

However, this power comes with complexity. We'll explore when it's worth it, simpler alternatives, and practical recommendations.

---

## 1. The Motivation: Why Cell-Level Reliability Matters

### The Problem with Global Weights

Traditional ensemble methods assign **global weights** to classifiers:
$$\hat{p}_i = \sum_{u=1}^m w_u \cdot r_{ui}$$

Problems:
- Classifier $u$ might be excellent on certain data regions but poor on others
- No adaptation to instance-specific patterns
- Treats all predictions from a classifier equally

### The Problem with Pure Reconstruction

Matrix factorization with uniform confidence:
$$L = \sum_{u,i} (r_{ui} - x_u^\top y_i)^2$$

Problems:
- All probabilities weighted equally
- Systematic errors reconstructed faithfully
- No notion of which cells are "signal" vs "noise"

### The Insight: Trust Varies Per Cell

**Key observation**: Reliability is a function of both classifier $u$ **and** instance $i$:

| Instance Type | Classifier A | Classifier B | Classifier C |
|--------------|--------------|--------------|--------------|
| Common cases | ✅ Reliable | ✅ Reliable | ✅ Reliable |
| Rare subtype X | ❌ Overconfident | ✅ Good | ❌ Uncertain |
| High-dimensional | ✅ Good | ❌ Poor calibration | ✅ Good |
| Noisy features | ❌ Overfits | ✅ Robust | ❌ Sensitive |

**We need cell-level $(u,i)$ reliability modeling, not just classifier-level or instance-level.**

---

## 2. Polarity Labels: What Are We Predicting?

### Definition

For a labeled point $i \in \mathcal{L}$ with true label $y_i \in \{0,1\}$ and classifier $u$'s probability $r_{ui}$:

**Step 1**: Convert probability to binary prediction using threshold $\tau$ (typically 0.5):
$$\hat{y}_{ui} = \mathbb{1}[r_{ui} \geq \tau]$$

**Step 2**: Determine polarity based on $(\hat{y}_{ui}, y_i)$:

$$p_{ui} = \begin{cases}
\text{TP} & \text{if } \hat{y}_{ui} = 1 \text{ and } y_i = 1 \\
\text{TN} & \text{if } \hat{y}_{ui} = 0 \text{ and } y_i = 0 \\
\text{FP} & \text{if } \hat{y}_{ui} = 1 \text{ and } y_i = 0 \\
\text{FN} & \text{if } \hat{y}_{ui} = 0 \text{ and } y_i = 1
\end{cases}$$

### The Training Data Explosion

For $|\mathcal{L}|$ labeled points and $m$ classifiers:
- **Traditional**: $|\mathcal{L}|$ supervised examples
- **Cell-level**: $m \times |\mathcal{L}|$ supervised examples

**Example**: With 10 classifiers and 500 labeled points, you get **5,000 training examples** for the polarity model.

This is **massive supervision** for learning which classifiers are reliable where.

---

## 3. What Can a Polarity Model Learn?

### Feature Space for Cell Reliability

For each cell $(u,i)$, we can extract features:

**Classifier-specific features**:
- Overall accuracy of classifier $u$
- Calibration metrics (Brier score)
- Model type and hyperparameters

**Instance-specific features**:
- Input features $x_i$ (if available)
- Difficulty indicators (ensemble variance)
- Distance to training centroids

**Cell-specific features** (most powerful):
- Probability value $r_{ui}$ itself
- Distance from decision boundary: $|r_{ui} - 0.5|$
- Agreement with other classifiers: $|r_{ui} - \bar{r}_i|$
- Entropy of classifier's predictions: $H_u$

### Patterns the Model Can Discover

A well-trained polarity model learns:

1. **Specialization**: "Classifier A is reliable when feature $X > 0.7$"
2. **Overconfidence**: "Classifier B's predictions $> 0.9$ are often FPs"
3. **Underconfidence**: "Classifier C's predictions around 0.6 are actually very reliable"
4. **Complementarity**: "When classifiers A and B disagree, trust A on subtype X, B on subtype Y"

This is essentially **mixture-of-experts gating** learned from data.

---

## 4. The Polarity Model as Cell-Level Supervision

### Formulation

Train a model $h$ that predicts cell reliability:
$$w_{ui} = h(\text{features}(u, i, r_{ui}, x_i))$$

where $w_{ui} \in [0,1]$ represents trust in cell $(u,i)$.

### Training on Labeled Data

For $i \in \mathcal{L}$, we know the true label $y_i$, so we can compute:

**Binary correctness**:
$$t_{ui} = \mathbb{1}[\hat{y}_{ui} = y_i]$$

**Continuous correctness** (often better):
$$t_{ui} = 1 - |r_{ui} - y_i|$$

**Training objective**:
$$\min_h \sum_{i \in \mathcal{L}} \sum_{u=1}^m L(t_{ui}, w_{ui})$$

where $L$ is squared error or binary cross-entropy.

### Integration with CF-Ensemble

Use learned $w_{ui}$ as confidence weights in reconstruction:

$$L_{\text{recon}} = \sum_{u,i} w_{ui} (r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2)$$

**Interpretation**: Reconstruction focuses on cells the polarity model deems reliable.

---

## 5. The Test Set Challenge: The Pseudo-Label Problem

### The Fundamental Issue

**Problem**: To compute polarity on test points $i \in \mathcal{U}$, we need $y_i$, which we don't have.

**Naive solution**: Use predicted label $\tilde{y}_i$ from ensemble.

**The danger**:
$$\tilde{p}_{ui} = \text{Polarity}(r_{ui}, \tilde{y}_i)$$

This creates a **circular dependency**:
1. Polarity model predicts which cells are reliable based on $\tilde{y}_i$
2. $\tilde{y}_i$ is computed by aggregating cells weighted by polarity
3. System believes its own guesses

### The Feedback Loop

```
Initial predictions → Pseudo-labels → Polarity estimates → 
Weighted aggregation → Updated predictions → Updated pseudo-labels → ...
```

**Risk**: If pseudo-labels are wrong in a structured way (e.g., systematically mislabeling a subpopulation), the feedback loop can **amplify errors** rather than correct them.

This is the classic **confirmation bias** problem in semi-supervised learning.

---

## 6. Principled Approach: Soft Pseudo-Labels and EM

### Treating Labels as Latent Variables

The correct framework is **Expectation-Maximization (EM)**:
- True labels $y_i$ for $i \in \mathcal{U}$ are **latent variables**
- We observe probabilities $r_{ui}$
- We want to jointly infer labels and learn reliability

### EM Algorithm for Polarity Model

**E-step** (Expectation): Compute soft pseudo-labels
$$q(y_i = 1) = \text{softmax}\left( \sum_u w_{ui} \cdot \log \frac{r_{ui}}{1-r_{ui}} \right)$$

Use **soft probabilities**, not hard labels, to avoid brittle feedback.

**M-step** (Maximization): Update polarity model and CF factors
$$\min_{X,Y,h} \rho \sum_{u,i} w_{ui}(r_{ui} - x_u^\top y_i)^2 + (1-\rho) \sum_{i \in \mathcal{L}} \text{CE}(y_i, \hat{p}_i) + \gamma \sum_{i \in \mathcal{U}} \text{CE}(q(y_i), \hat{p}_i)$$

where $\gamma < 1$ is a **confidence penalty** on pseudo-labels (e.g., $\gamma = 0.1$).

### Regularization Strategies

1. **Soft labels only**: Never convert $q(y_i)$ to hard 0/1
2. **Entropy regularization**: Penalize overconfident pseudo-labels
3. **Conservative updates**: Small learning rate on test set weights
4. **Warm-up**: Train on labeled data only for several epochs first

---

## 7. When Polarity Models Add Too Much Complexity

### Four Design Burdens

**1. Threshold dependence**: Choice of $\tau$ affects TP/FP/TN/FN split
- Different classifiers may need different thresholds
- Binary polarity loses information about confidence levels

**2. Calibration mismatch**: Base models' probabilities may not be comparable
- One model's 0.7 ≠ another model's 0.7
- Requires pre-calibration step (Platt scaling, isotonic regression)

**3. Distribution shift**: Labeled region $\neq$ unlabeled region
- Polarity model trained on $\mathcal{L}$ may not generalize to $\mathcal{U}$
- Features predictive of reliability on train may differ on test

**4. Circularity**: Polarity depends on $\tilde{y}$, which depends on aggregation, which depends on polarity
- Hard to debug when something goes wrong
- Instability can arise from feedback loops

### When It's Not Worth It

Polarity models are **overkill** if:
- ✗ Base models are already highly correlated (low diversity)
- ✗ Probabilities are poorly calibrated across models
- ✗ Limited labeled data ($|\mathcal{L}| < 100$)
- ✗ Simple stacking already works well
- ✗ Can't control feedback loop (no soft labels, uncertainty)

---

## 8. Simpler Alternatives That Capture 80% of Value

### Option A: Direct Reliability Weighting (Recommended)

**Idea**: Learn continuous reliability weights without explicit TP/FP/TN/FN classification.

**Model**:
$$w_{ui} = h(r_{ui}, u, i, x_i) \in [0,1]$$

**Training** (on labeled data only):
$$\min_h \sum_{i \in \mathcal{L}} \sum_{u=1}^m (t_{ui} - w_{ui})^2$$

where $t_{ui} = 1 - |r_{ui} - y_i|$ (continuous correctness).

**Advantages**:
- ✅ No threshold needed
- ✅ No pseudo-labels on test
- ✅ Directly usable as $C$ in CF objective
- ✅ Trained only on labeled data (no circularity)

**Test-time usage**: Compute $w_{ui}$ for test points using $h$, **without needing $y_i$**.

---

### Option B: Instance-Dependent Gating (Mixture of Experts)

**Idea**: For each instance, learn which classifiers to trust.

**Model**:
$$\alpha_{ui} = \text{softmax}_u(g(r_{ui}, u, i, x_i))$$
$$\hat{p}_i = \sum_{u=1}^m \alpha_{ui} \cdot r_{ui}$$

where $\sum_u \alpha_{ui} = 1$.

**Training**:
$$\min_g \sum_{i \in \mathcal{L}} \text{CE}(y_i, \hat{p}_i)$$

**Advantages**:
- ✅ Input-dependent weighting (stronger than global weights)
- ✅ Can be regularized with low-rank structure if needed
- ✅ No explicit TP/FP/TN/FN needed
- ✅ Standard stacking with attention mechanism

---

### Option C: Stay with KD-Style Combined Objective

**Idea**: The combined reconstruction + supervised objective already addresses signal-noise separation.

**Reminder**:
$$\mathcal{L} = \rho \cdot L_{\text{recon}}(X,Y) + (1-\rho) \cdot L_{\text{sup}}(X,Y,\theta)$$

with simple confidence weights (e.g., $c_{ui} = |r_{ui} - 0.5|$).

**Advantages**:
- ✅ Minimum complexity upgrade over pure MF
- ✅ Already incorporates supervision
- ✅ No additional models to train
- ✅ Proven approach from knowledge distillation

**When it's enough**: If base models are reasonably diverse and well-calibrated, this may be all you need.

---

## 9. Mathematical Formulation: Option A in Detail

### Complete Reliability Weight Model

**Architecture**: 
$$w_{ui} = \sigma(f_\phi(z_{ui}))$$

where:
- $z_{ui} = [r_{ui}, |r_{ui} - 0.5|, |r_{ui} - \bar{r}_i|, \text{features}_u, \text{features}_i]$ (feature vector)
- $f_\phi$ is a neural network or gradient boosting model
- $\sigma$ is sigmoid to ensure $w_{ui} \in [0,1]$

**Training objective**:
$$\min_\phi \sum_{i \in \mathcal{L}} \sum_{u=1}^m \left[ (t_{ui} - w_{ui})^2 + \beta \cdot \text{entropy}(w_{ui}) \right]$$

where:
- $t_{ui} = 1 - |r_{ui} - y_i|$ is continuous correctness
- Entropy term $-w_{ui} \log w_{ui} - (1-w_{ui})\log(1-w_{ui})$ prevents overconfident weights

**Integration with CF-Ensemble**:
$$\mathcal{L}_{\text{full}} = \rho \sum_{u,i} w_{ui}(r_{ui} - x_u^\top y_i)^2 + \lambda(\|X\|_F^2 + \|Y\|_F^2) + (1-\rho) \sum_{i \in \mathcal{L}} \text{CE}(y_i, g_\theta(\hat{r}_{\cdot i}))$$

---

## 10. When Polarity Models Are Worth the Complexity

### Ideal Conditions

Polarity models provide **significant value** when:

✅ **High base model diversity**: 
- Many classifiers ($m \geq 10$)
- Different architectures, features, hyperparameters
- Clear complementarity (each excels in different regions)

✅ **Sufficient labeled data**:
- $|\mathcal{L}| \times m \geq 1000$ (cell-level examples)
- Diverse coverage of data distribution

✅ **Stable pseudo-label generation**:
- Can use soft probabilities for test set
- EM algorithm with conservative updates
- Can afford computational cost of iterations

✅ **Transductive setting**:
- Test set known at training time
- Can evaluate fairly against baselines with same access

✅ **Heterogeneous errors**:
- Base models make different types of mistakes
- Some models overfit, others underfit
- Regional specialization exists

---

### Example Use Case

**Medical diagnosis with multi-modal data**:
- **Classifiers**: Imaging CNN, lab value tree model, clinical notes LSTM, demographic LR, hybrid ensemble
- **Patterns**: 
  - CNN excels on visual pathologies but fails on lab-only cases
  - Lab model good for metabolic conditions but misses imaging findings
  - LSTM captures longitudinal patterns but overfits rare diseases
- **Benefit**: Polarity model learns which modality to trust for each patient type

**Result**: Cell-level reliability weighting significantly outperforms global weighting or uniform confidence.

---

## 11. Practical Implementation Strategy

### Phase 1: Start Simple (Week 1-2)

1. **Baseline**: CF-Ensemble with KD-style combined objective
   - Use simple confidence: $c_{ui} = |r_{ui} - 0.5|$ or label-aware
   - Establish baseline performance

2. **Quick win**: Label-aware confidence on training set
   ```python
   for i in labeled_idx:
       if y[i] == 1:
           C[:, i] = R[:, i]  # Reward high predictions
       else:
           C[:, i] = 1 - R[:, i]  # Reward low predictions
   ```

### Phase 2: Add Reliability Weighting (Week 3-4)

3. **Train reliability model** (Option A):
   ```python
   # Features
   features = [
       R.flatten(),  # Probabilities
       np.abs(R - 0.5).flatten(),  # Distance from 0.5
       np.tile(classifier_accuracy, n),  # Per-classifier calibration
   ]
   
   # Targets (on labeled data only)
   targets = 1 - np.abs(R[:, labeled] - y[labeled])
   
   # Train
   model = GradientBoostingRegressor()
   model.fit(features, targets.flatten())
   
   # Predict weights for all cells
   W = model.predict(all_features).reshape(m, n)
   W = np.clip(W, 0.1, 1.0)  # Floor at 0.1
   ```

4. **Use as confidence in CF**:
   ```python
   trainer = CFEnsembleTrainer(rho=0.5)
   ensemble_data = EnsembleData(R, labels, C=W)
   trainer.fit(ensemble_data)
   ```

### Phase 3: Full Polarity Model (Week 5-6) - Optional

5. **Implement EM with soft pseudo-labels**:
   - Only if Phase 2 shows insufficient improvement
   - Only if you have computational resources for iterations
   - Only if you can carefully tune regularization

---

## 12. Evaluation and Diagnostics

### Metrics to Track

**Reliability model quality**:
- **Correlation**: Between $w_{ui}$ and $t_{ui}$ on held-out labeled data
- **AUC**: Treating $w_{ui}$ as prediction of binary correctness $\mathbb{1}[|\hat{y}_{ui} - y_i| = 0]$

**CF-Ensemble performance**:
- **With vs without reliability weights**: Compare ROC-AUC
- **Weight entropy**: $H = -\sum_{u,i} w_{ui} \log w_{ui}$
  - Too low → overconfident (possibly overfitting)
  - Too high → not selective enough (close to uniform)

**Visualization**:
- **Heatmap** of $W$: Do high-weight cells cluster meaningfully?
- **Per-classifier distributions**: $w_{u\cdot}$ across instances
- **Per-instance distributions**: $w_{\cdot i}$ across classifiers

---

## 13. Comparison: Simple vs. Advanced Confidence

| Approach | Complexity | Training Data | Test Labels Needed | Performance Gain |
|----------|------------|---------------|-------------------|------------------|
| **Uniform** | Minimal | None | No | Baseline |
| **Certainty** ($\|r-0.5\|$) | Minimal | None | No | Small (+2%) |
| **Label-aware** (train only) | Low | $\mathcal{L}$ | No | Moderate (+5%) |
| **Reliability model** | Medium | $m \times \mathcal{L}$ | No | Large (+8-12%) |
| **Full polarity + EM** | High | $m \times \mathcal{L}$ + $\mathcal{U}$ | Pseudo | Large? (+10-15%) |

**Recommendation**: Start with **reliability model** (Option A). It captures most of the value with manageable complexity.

---

## 14. Code Template: Reliability Weight Model

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class ReliabilityWeightModel:
    """Learn cell-level reliability weights from labeled data."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42
        )
    
    def extract_features(self, R, classifier_stats=None):
        """
        Extract features for each cell (u, i).
        
        Parameters:
        - R: Probability matrix (m × n)
        - classifier_stats: Optional dict with per-classifier metrics
        
        Returns:
        - features: (m*n × d) feature matrix
        """
        m, n = R.shape
        
        features = []
        
        # Cell-specific features
        features.append(R.flatten())  # Raw probability
        features.append(np.abs(R - 0.5).flatten())  # Distance from threshold
        
        # Instance agreement
        R_mean = np.mean(R, axis=0)  # (n,)
        R_std = np.std(R, axis=0)  # (n,)
        features.append(np.repeat(R_mean, m))  # Broadcast to (m*n,)
        features.append(np.repeat(R_std, m))
        
        # Classifier statistics
        if classifier_stats is not None:
            for stat_name, stat_values in classifier_stats.items():
                features.append(np.tile(stat_values, n))
        
        return np.column_stack(features)
    
    def fit(self, R, labels, labeled_idx):
        """
        Train reliability model on labeled data.
        
        Parameters:
        - R: Probability matrix (m × n)
        - labels: Ground truth (n,) with NaN for unlabeled
        - labeled_idx: Boolean mask for labeled points
        """
        m, n = R.shape
        
        # Extract features for labeled cells only
        R_labeled = R[:, labeled_idx]
        y_labeled = labels[labeled_idx]
        
        # Compute features
        features_all = self.extract_features(R)
        
        # Mask for labeled cells
        labeled_cell_mask = np.repeat(labeled_idx, m)
        features_labeled = features_all[labeled_cell_mask]
        
        # Compute targets: continuous correctness
        targets = 1 - np.abs(R_labeled - y_labeled).flatten()
        
        # Train
        self.model.fit(features_labeled, targets)
        
        return self
    
    def predict(self, R, classifier_stats=None):
        """
        Predict reliability weights for all cells.
        
        Parameters:
        - R: Probability matrix (m × n)
        - classifier_stats: Optional classifier statistics
        
        Returns:
        - W: Reliability weights (m × n)
        """
        features = self.extract_features(R, classifier_stats)
        weights = self.model.predict(features)
        
        # Clip to [0.1, 1.0] to avoid zero weights
        weights = np.clip(weights, 0.1, 1.0)
        
        return weights.reshape(R.shape)

# Usage
rel_model = ReliabilityWeightModel()
rel_model.fit(R, labels, labeled_idx)
W = rel_model.predict(R)

# Use W as confidence in CF-Ensemble
ensemble_data = EnsembleData(R, labels, C=W)
trainer = CFEnsembleTrainer(rho=0.5)
trainer.fit(ensemble_data)
```

---

## 15. Research Directions and Open Questions

### Theoretical Questions

1. **Generalization bounds**: How does reliability model generalization affect CF-Ensemble performance?
2. **Identifiability**: Are learned weights unique? Can we distinguish true reliability from random correlation?
3. **Sample complexity**: How many labeled examples needed per cell type?

### Algorithmic Improvements

1. **Multi-task learning**: Share reliability model across related datasets
2. **Active learning**: Which cells should we label to maximally improve weight estimates?
3. **Online updates**: Adapt weights as new data arrives

### Extensions

1. **Regression**: Extend from binary to continuous targets
2. **Multi-class**: Polarity for K-way classification
3. **Structured outputs**: Sequence, graph, or image prediction

---

## 16. Connection to Related Work

### Mixture of Experts (MoE)

Polarity models are **learned gating** in MoE:
- **Gating network**: $g(x_i) \to \alpha_{ui}$ (which expert to trust)
- **Polarity model**: $h(r_{ui}, x_i) \to w_{ui}$ (cell-level trust)

**Difference**: Polarity uses base model outputs $r_{ui}$ as features, MoE uses inputs $x_i$.

### Stacking with Meta-Features

Standard stacking:
$$\hat{p}_i = g([r_{1i}, \ldots, r_{mi}])$$

Polarity-aware stacking:
$$\hat{p}_i = g([r_{1i}, \ldots, r_{mi}, w_{1i}, \ldots, w_{mi}])$$

**Benefit**: Meta-learner sees both predictions and reliability estimates.

### Calibration and Uncertainty

Polarity model can incorporate:
- **Platt scaling**: Pre-calibrate base models
- **Epistemic uncertainty**: Model prediction variance
- **Aleatoric uncertainty**: Inherent label noise

---

## 17. Decision Framework: Should You Use Polarity Models?

### Decision Tree

```
START: Do you have diverse base models (m ≥ 10)?
├─ NO → Stick with simple confidence (label-aware or certainty)
└─ YES → Do you have sufficient labeled data (m×|L| ≥ 1000)?
    ├─ NO → Use label-aware confidence on training set
    └─ YES → Is there clear complementarity in errors?
        ├─ NO → Try reliability model, but may not help much
        └─ YES → Can you afford computational cost?
            ├─ NO → Use reliability model (Option A)
            └─ YES → Consider full polarity + EM (Option C)
```

---

## 18. Pragmatic Recommendation

### The Goldilocks Approach

**Too simple**: Uniform confidence $c_{ui} = 1$
- Misses opportunity to emphasize reliable cells

**Just right**: **Reliability weight model** (Option A)
- Cell-level supervision ($m \times |\mathcal{L}|$ examples)
- No pseudo-labels needed
- Directly integrates with CF-Ensemble
- Trained only on labeled data (stable)

**Too complex**: Full polarity model with EM
- Pseudo-label feedback loops
- Calibration challenges
- Debugging difficulty
- Marginal gains over reliability model

---

## 19. Implementation Priority

### Recommended Sequence

**Phase 1** (MVP - Week 4):
1. ✅ CF-Ensemble with simple confidence
2. ✅ Establish baseline performance

**Phase 2** (First enhancement - Week 5-6):
3. ✅ Add reliability weight model (Option A)
4. ✅ Compare with baseline
5. ✅ If +5% improvement or more → success

**Phase 3** (Optional - Week 7+):
6. ⚠️ Only if Phase 2 insufficient
7. ⚠️ Full polarity model with soft pseudo-labels
8. ⚠️ Carefully tune EM algorithm

---

## 20. Summary

### Key Takeaways

1. **Cell-level reliability** is the missing ingredient in pure matrix factorization
2. **Massive supervision**: $m \times |\mathcal{L}|$ training examples available
3. **Simpler is often better**: Reliability weights (Option A) capture 80% of value
4. **Full polarity models** add complexity that may not be worth the gain
5. **No pseudo-labels on test**: Avoid circular dependencies when possible

### The Core Insight

> **Trust varies per (classifier, instance) pair. Learn that trust from labeled data. Use it to weight reconstruction.**

This is the pragmatic evolution of the polarity model idea: keep the "massive supervision" benefit, avoid the "pseudo-label hallucination" risk.

---

## 21. Conclusion

The polarity model concept—learning which cells in the probability matrix are reliable—is powerful. However, the **full TP/FP/TN/FN prediction on test data** introduces complexity and instability through pseudo-labels.

The **reliability weight model** (Option A) captures the essential insight:
- ✅ Cell-level supervision from labeled data
- ✅ Direct integration with CF-Ensemble
- ✅ No circular dependencies
- ✅ Manageable complexity

**Start here**. Only escalate to full polarity models if evidence shows it's necessary.

Your instinct was correct: "this makes CF ensemble learning too complex." The reliability weight variant gives you the power without the pain.

---

## References

1. Jacobs, R., et al. (1991). *Adaptive Mixtures of Local Experts*. Neural Computation.
2. Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*. CRC Press.
3. Blum, A., & Mitchell, T. (1998). *Combining Labeled and Unlabeled Data with Co-Training*. COLT.

---

**Next**: See [Implementation Roadmap](../../IMPLEMENTATION_ROADMAP.md) for how to add reliability weighting to your CF-Ensemble pipeline.
