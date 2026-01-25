# Failure Mode: Misunderstanding Transductive Learning in CF-Ensemble

**Category:** Algorithmic Misuse  
**Severity:** Critical (Causes complete failure)  
**Date Identified:** 2026-01-25

---

## TL;DR

**Problem:** Treating CF-Ensemble like a traditional classifier with separate train/test splits breaks the transductive learning assumption.

**Solution:** Train on ALL data (train + test) with test labels masked, then use learned latent factors for prediction (not cold-start computation).

---

## The Problem: Wrong Train-Test Split

### What We Did (WRONG)

```python
# Traditional ML approach - DOES NOT WORK for CF-Ensemble
R_train = R[:, labeled_idx]
y_train = labels[labeled_idx]

# Train on training data only
ensemble_data = EnsembleData(R_train, y_train)
trainer.fit(ensemble_data)

# Predict on separate test set (cold-start)
y_pred = trainer.predict(R_new=R_test)  # ❌ WRONG!
```

**Result:** 
- PR-AUC: 0.02-0.11 (worse than random!)
- Performance worse than simple averaging
- No convergence after 100-200 iterations

### Why This Fails

CF-Ensemble is fundamentally **transductive**, not inductive:

1. **Base classifiers have ALREADY made predictions** on all data (train + test)
2. **The probability matrix R includes test instances**
3. **We want to learn how to aggregate these existing predictions**
4. **Cold-start prediction throws away this information**

This is like asking a recommender system to predict ratings for users it has never seen, when it actually *has* seen their rating behavior - we just masked some ratings!

---

## The Recommender System Analogy

### Standard Recommender System

| Concept | Recommender System | CF-Ensemble |
|---------|-------------------|-------------|
| "Users" | Actual users | Base classifiers |
| "Items" | Products/movies | Data instances |
| "Ratings" | User preferences (1-5 stars) | Predicted probabilities [0,1] |
| **Goal** | Predict missing ratings | Predict masked labels |
| **Metric** | RMSE on held-out ratings | PR-AUC on held-out labels |

### Key Insight from Recommender Systems

In a recommender system with matrix factorization:

**Scenario 1: Warm Start (Item seen during training)**
```python
# Item i was in training set (some users rated it)
# Use its LEARNED latent factor
y_i = Y[:, i]  # Already optimized during training
rating_prediction = X.T @ y_i
```

**Scenario 2: Cold Start (New item, never seen)**
```python
# Item i_new is completely new
# Must compute its factor from scratch
y_i_new = (X^T X + λI)^{-1} X^T r_i_new
rating_prediction = X.T @ y_i_new
```

### CF-Ensemble is (Usually) Warm Start!

**The critical realization:**
- Test instances in CF-Ensemble are **NOT new items**
- Base classifiers have **already predicted** on them
- We have their probability vectors in R
- They were **present during training** (just with masked labels)
- This is **warm start**, not cold start!

**Using cold start prediction is like:**
- A recommender system that has seen 1000 users rate a movie
- But then throws away all those learned patterns
- And recomputes the movie's factor from scratch for each prediction
- Obviously wasteful and suboptimal!

---

## The Correct Solution: Transductive Learning

### What We Should Do (CORRECT)

```python
# Combine ALL data (train + test), mask test labels
R_combined = np.hstack([R_train, R_test])
labels_combined = np.concatenate([
    y_train,
    np.full(len(y_test), np.nan)  # Masked, not missing!
])

# Train on ALL data - learn factors for everything
ensemble_data = EnsembleData(R_combined, labels_combined)
trainer.fit(ensemble_data)

# Use LEARNED factors for test predictions (warm start)
all_predictions = trainer.predict()  # ✅ CORRECT
y_pred_test = all_predictions[len(y_train):]
```

**Result:**
- Actually uses the information available
- Test instances get optimized latent factors
- Aggregator learns from full probability structure

### Mathematical Justification

During training, we optimize:

$$\min_{X, Y, \theta} \sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 + \sum_{i \in \mathcal{L}} \text{CE}(y_i, g_\theta(\hat{r}_{\cdot i}))$$

where:
- $\mathcal{L}$ = labeled instances (train)
- $\mathcal{U}$ = unlabeled instances (test)

**Crucially:**
- **Both** $\mathcal{L}$ and $\mathcal{U}$ contribute to the first term (reconstruction)
- **Only** $\mathcal{L}$ contributes to the second term (supervision)

This means:
- Test instance factors $Y_{\mathcal{U}}$ are learned from reconstruction + similarity to train instances
- They benefit from the low-rank structure and learned classifier factors
- Cold-start computation ignores this rich information!

---

## Performance Comparison

### On Imbalanced Data (10% Positive)

| Method | Approach | PR-AUC | Converged? |
|--------|----------|--------|-----------|
| Simple Average | N/A | 0.285 | N/A |
| Stacking | Inductive | 0.522 | ✓ |
| **CF-Ensemble (Wrong)** | **Cold-start inductive** | **0.136** | **✗** |
| CF-Ensemble (Fixed) | Transductive | TBD | TBD |

**The wrong approach:**
- ❌ Worse than simple averaging
- ❌ Worse than random (baseline PR-AUC for 10% positive ≈ 0.10)
- ❌ Never converges

---

## When to Use Cold-Start vs. Warm-Start

### Use Warm-Start (Transductive) When:

✅ **Test instances are known at training time**
- You have their feature vectors
- Base models have made predictions on them
- You're aggregating existing predictions
- **This is the standard CF-Ensemble setting**

**Example:** 
- Biomedical prediction: You have predictions from 10 algorithms on 1000 patients
- Goal: Aggregate them well, using subset with known outcomes for training

### Use Cold-Start (Inductive) When:

✅ **Test instances arrive AFTER training**
- Truly new data that base models haven't seen
- Need to make predictions on-the-fly
- Can't retrain for each new instance

**Example:**
- Real-time system: New patient arrives, need immediate prediction
- Base models make predictions, need to aggregate them instantly
- Must use: $y_{\text{new}} = (X^\top X + \lambda I)^{-1} X^\top r_{\text{new}}$

### CF-Ensemble is Mostly Transductive

**In most practical scenarios:**
- Batch prediction setting (not online)
- Have all base model predictions upfront
- Can train with test instances present
- **Use transductive learning (warm start)**

**Only use inductive (cold start) when:**
- Truly cannot include test data in training
- Real-time constraints require immediate prediction
- Distribution shift makes transductive learning unreliable

---

## The Amazon Recommender System Perspective

### How would an Amazon ML scientist approach this?

**Standard recommender problem:**
> "Given user-item rating matrix with missing entries, predict missing ratings"

**Solution:** Matrix factorization learns user factors $X$ and item factors $Y$ such that $R \approx X^\top Y$

**For new predictions:**
- **Seen users, seen items:** Use learned factors (warm start)
- **New user, seen items:** Compute user factor from their ratings
- **Seen user, new item:** Compute item factor from its ratings
- **New user, new item:** Use cold start methods (content features, etc.)

**CF-Ensemble adds supervision:**
> "Not only predict ratings well, but ensure aggregated predictions match ground truth labels"

This is like Amazon also caring that:
- High predicted ratings correlate with actual purchases
- Aggregated ratings predict customer satisfaction
- The reconstruction quality + predictive accuracy tradeoff

**Amazon would:**
1. Use **transductive learning** for batch settings (warm start)
2. Use **content features** or **deep learning** for cold start
3. **Never throw away** learned factors when they're available!
4. Optimize for **both** reconstruction AND business metric (in our case, label accuracy)

---

## Implementation Notes

### Detecting the Issue

**Red flags that you're doing it wrong:**
- Performance worse than simple averaging
- No convergence after many iterations
- Test error >>> Train error (not generalization, but cold-start penalty)
- Predictions seem random or biased

**Code smells:**
```python
# BAD: Training on subset of columns
R_train = R[:, train_idx]
trainer.fit(EnsembleData(R_train, y_train))

# BAD: Using R_new for test predictions
y_pred = trainer.predict(R_new=R_test)

# BAD: Separate train/test matrices
ensemble_data_train = EnsembleData(R_train, y_train)
ensemble_data_test = EnsembleData(R_test, None)  # Wrong!
```

### Correct Implementation

```python
# GOOD: Single matrix with masked labels
n_train = len(y_train)
n_test = len(y_test)
R_all = np.hstack([R_train, R_test])
labels_all = np.concatenate([y_train, np.full(n_test, np.nan)])

# GOOD: Train on everything
ensemble_data = EnsembleData(R_all, labels_all)
trainer.fit(ensemble_data)

# GOOD: Use learned factors
all_preds = trainer.predict()  # No R_new argument
y_pred_test = all_preds[n_train:]  # Extract test portion

# GOOD: Alternatively, for truly new data
if new_data_arrives:
    y_pred_new = trainer.predict(R_new=R_new_data)  # Cold start OK here
```

---

## Lessons Learned

### 1. Not All ML is Inductive

- **Inductive:** Learn from train, generalize to unseen test
- **Transductive:** Have access to test inputs (not labels) during training
- **CF-Ensemble is transductive** by design

### 2. Base Model Predictions are Data

- In traditional ML: test features are inputs
- In CF-Ensemble: test **predictions** are inputs
- We already have them at training time!
- Use them!

### 3. Recommender System Intuition Helps

- Think: "How would Netflix predict ratings for a movie in their database?"
- Answer: Use its learned latent factors (warm start)
- Not: Recompute factors from scratch each time (cold start)
- CF-Ensemble same principle

### 4. Read Your Own Documentation

The theory document (`docs/methods/cf_ensemble_optimization_objective_tutorial.md`) Section 6 clearly states:

> "Crucially, **both sets are used during training**, but labels are only available for $\mathcal{L}$. This is **transductive** or **semi-supervised** learning."

We just didn't implement it correctly! 🤦

---

## Related Failure Modes

See also:
- `optimization_instability.md` - Why alternating ALS + GD doesn't converge
- `confidence_weights.md` - Importance of label-aware confidence
- `hyperparameter_sensitivity.md` - Tuning latent_dim, lambda_reg, rho

---

## References

1. **Transductive Learning:**
   - Vapnik, V. (1998). *Statistical Learning Theory*. Chapter on transduction.
   - Zhou, D., et al. (2004). "Learning with Local and Global Consistency." NeurIPS.

2. **Matrix Factorization for Recommender Systems:**
   - Koren, Y., et al. (2009). "Matrix Factorization Techniques for Recommender Systems." IEEE Computer.
   - Hu, Y., et al. (2008). "Collaborative Filtering for Implicit Feedback Datasets." ICDM.

3. **Cold Start Problem:**
   - Schein, A., et al. (2002). "Methods and Metrics for Cold-Start Recommendations." SIGIR.
   - Sedhain, S., et al. (2014). "Social Collaborative Filtering for Cold-start Recommendations." RecSys.

---

**Key Takeaway:** CF-Ensemble is a transductive method pretending to be inductive will fail catastrophically. Always train on all data with masked test labels!
