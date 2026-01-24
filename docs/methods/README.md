# CF-Ensemble Methods Documentation

This directory contains comprehensive documentation of the CF-Ensemble methodology, optimization objectives, and theoretical foundations.

---

## Core Tutorials (Start Here)

### 1. [Knowledge Distillation Tutorial](knowledge_distillation_tutorial.md)
**Foundation concept that inspired the CF-Ensemble approach**

Learn how knowledge distillation combines soft targets (teacher imitation) and hard labels (ground truth) to train effective student models. Understanding this is crucial for grasping the CF-Ensemble optimization objective.

**Key concepts**:
- Soft vs hard targets
- Temperature-scaled softmax
- The $T^2$ correction factor
- Why combining imitation and supervision works

**Time**: ~30 minutes

---

### 2. [CF-Ensemble Optimization Objective](cf_ensemble_optimization_objective_tutorial.md)
**The complete mathematical framework for CF-based ensemble learning**

Discover how knowledge distillation principles generalize to ensemble learning through collaborative filtering. This tutorial develops the unified objective that combines matrix reconstruction with supervised learning.

**Key concepts**:
- Probability matrix as ensemble knowledge
- Matrix factorization for latent structure
- The combined loss: $\mathcal{L} = \rho \cdot L_{\text{recon}} + (1-\rho) \cdot L_{\text{sup}}$
- Why this should work better than pure reconstruction
- Transductive learning for ensembles

**Time**: ~45 minutes

---

### 3. [Confidence Weighting & Reliability Learning](confidence_weighting/)
**From global reconstruction to fine-grained trust**

A complete subsection on confidence weighting strategies and learned reliability weights.

**Documents**:
- **[Base Classifier Quality Analysis](confidence_weighting/base_classifier_quality_analysis.md)** 🎯 NEW
  - When does confidence weighting help?
  - Quality thresholds (60-85% sweet spot)
  - Debugging poor performance
  - Time: ~30 minutes
  
- **[Polarity Models Tutorial](confidence_weighting/polarity_models_tutorial.md)**
  - Cell-level reliability learning
  - Learned vs fixed confidence strategies
  - Implementation guide
  - Time: ~40 minutes

**Key concepts**:
- Cell-level vs global confidence weighting
- Massive supervision: $m \times |\mathcal{L}|$ training examples
- Quality-confidence relationship
- When confidence weighting is (and isn't) effective

**Implementation priority**: Phase 3 complete ✅

---

## Reading Order

For newcomers to the project:

```
1. Start: CF-Ensemble README.md (project overview)
2. Foundation: knowledge_distillation_tutorial.md (~30 min)
3. Core Method: cf_ensemble_optimization_objective_tutorial.md (~45 min)
4. Confidence Weighting:
   a. base_classifier_quality_analysis.md (~30 min) - When it works
   b. polarity_models_tutorial.md (~40 min) - How to implement
5. Practical: hyperparameter_tuning.md (~5-45 min, start with quick start)
6. Technical: als_vs_pytorch.md (~30 min, optional)
7. Math Deep-Dive: als_mathematical_derivation.md (~60 min, optional)
8. Quick Ref: QUICK_REFERENCE.md (5 min)
9. Implementation: notebooks/01_collaborative_filtering/
10. Advanced: Original research PDFs
```

**Total time to understand core concepts**: ~3-4 hours  
**Time to start experimenting**: ~10 minutes (quick start guides)

---

## Practical Guides

### 4. [Hyperparameter Tuning for CF-Ensemble](hyperparameter_tuning.md)
**How to determine ρ, d, and λ for your dataset**

Comprehensive guide to selecting and tuning hyperparameters, with special focus on the critical ρ parameter that balances reconstruction and supervision.

**Key concepts**:
- What is ρ and why it matters (most important hyperparameter!)
- Quick start defaults: ρ=0.5, d=20, λ=0.01
- Cross-validation for ρ selection
- When to use high vs low ρ
- Grid search and Bayesian optimization
- Adaptive ρ strategies (advanced)

**Includes**:
- Rule of thumb guidelines (few labels → high ρ, many labels → low ρ)
- Complete code examples for cross-validation
- Decision tree for quick troubleshooting
- Performance debugging checklist

**Time**: ~45 minutes (5 minutes for quick start, 45 for full guide)

**Must-read before**: Running experiments on real data

---

### 5. [ALS vs PyTorch Gradient Descent](als_vs_pytorch.md)
**Comparing optimization approaches: Closed-form vs Gradient-based**

Explains why ALS is state-of-the-art for matrix factorization, when to consider PyTorch, and how they should give equivalent results.

**Key concepts**:
- Why ALS is SoTA for collaborative filtering
- Advantages of closed-form updates (no learning rate, guaranteed convergence)
- When PyTorch is better (GPU, large scale, neural extensions)
- Mathematical equivalence (should converge to same solution)
- Implementation sketch of PyTorch version

**Includes**:
- Side-by-side comparison table
- Performance benchmarks (small/medium/large datasets)
- Code for PyTorch implementation
- Validation experiment (verify consistency)
- Hybrid approach (ALS init + PyTorch fine-tune)

**Time**: ~30 minutes

**Future work**: Phase 5+ may add PyTorch implementation for scalability

---

### 6. [ALS Mathematical Derivation](als_mathematical_derivation.md) ⭐
**Complete step-by-step derivation of the closed-form ALS updates**

NEW! Comprehensive mathematical derivation showing how we arrive at the ALS update equations. Essential reading for understanding the optimization algorithm.

**Key concepts**:
- Problem decomposition (per-classifier, per-instance)
- Gradient derivation from first principles
- Closed-form solution via setting gradient to zero
- Convergence properties and guarantees
- Computational complexity analysis

**Derives**:
- Classifier update: $x_u = (Y C_u Y^T + \lambda I)^{-1} Y C_u r_u$
- Instance update: $y_i = (X C_i X^T + \lambda I)^{-1} X C_i r_i$

**Includes**:
- Step-by-step algebraic manipulations
- Matrix calculus rules
- Numerical stability considerations
- Vectorization opportunities
- Exercises for self-study

**Time**: ~60 minutes (20 for quick scan, 60 for full understanding)

**Must-read before**: Implementing your own ALS solver or extending the algorithm

---

## Supporting Documents

### Historical Context

- **[cf_ensemble_optimization_objective.md](../../dev/methods/cf_ensemble_optimization_objective.md)** *(dev notes)*: Original Q&A-style notes that led to the breakthrough
- **[knowledge_distillation.md](../../dev/methods/knowledge_distillation.md)** *(dev notes)*: Initial exploration of KD concepts

*Note: dev/ documents are private development notes, not public documentation*

### Research Papers

- **[CF-EnsembleLearning-Intro.pdf](../CF-EnsembleLearning-Intro.pdf)**: Comprehensive introduction to the original CF-Ensemble concept
- **[CFEnsembleLearning-optimization.pdf](../CFEnsembleLearning-optimization.pdf)**: Detailed optimization formulation and ALS algorithm
- **[CF-based-ensemble-learning-slides.pdf](../CF-based-ensemble-learning-slides.pdf)**: Presentation slides

---

## Key Mathematical Objects

Quick reference for notation used throughout:

| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| $R$ | Probability matrix (base models × data points) | $m \times n$ |
| $r_{ui}$ | Classifier $u$'s probability for point $i$ | $[0,1]$ |
| $X$ | Classifier latent factors | $d \times m$ |
| $Y$ | Instance latent factors | $d \times n$ |
| $x_u$ | Latent vector for classifier $u$ | $\mathbb{R}^d$ |
| $y_i$ | Latent vector for data point $i$ | $\mathbb{R}^d$ |
| $\hat{r}_{ui}$ | Reconstructed probability $= x_u^\top y_i$ | $[0,1]$ |
| $C$ | Confidence/reliability weights | $m \times n$ |
| $\mathcal{L}$ | Labeled point indices | $\subseteq \{1,\ldots,n\}$ |
| $\mathcal{U}$ | Unlabeled point indices | $\subseteq \{1,\ldots,n\}$ |
| $\rho$ | Trade-off: reconstruction vs supervision | $[0,1]$ |
| $\lambda$ | Regularization strength | $\mathbb{R}_+$ |
| $d$ | Latent dimension | $\mathbb{N}$ |

---

## The Central Innovation

### Previous Approach (Failed)
```
Pure reconstruction: min ||R - XY^T||²
Problem: Faithfully reproduces base model errors
```

### New Approach (KD-Inspired)
```
Combined objective: L = ρ·L_recon + (1-ρ)·L_sup
Solution: Learns which patterns are signal vs noise
```

The key insight: **Adding supervised loss teaches the model what "signal" means**, preventing it from simply reproducing systematic errors in the base models.

---

## Implementation Status

### ✅ Completed
- Theoretical framework fully developed
- Mathematical formulation finalized
- Tutorial documentation written (3 comprehensive guides)
- Project structure reorganized
- Reliability weight model designed

### 🚧 In Progress (Week 1-4)
- Implementation of new combined objective
- Data structures and loss functions
- ALS optimization algorithm
- Experimental validation on synthetic data

### 📋 Planned (Week 5-8)
- Learned reliability weights (Phase 3 enhancement)
- Real-world dataset validation
- Comparison with stacking and boosting
- Extension to multi-class classification
- Non-linear variants (neural factorization)

---

## Quick Start for Researchers

If you're already familiar with collaborative filtering and want to dive straight into the method:

1. **Core equation**:
   $$\mathcal{L} = \rho \sum_{u,i} c_{ui}(r_{ui} - x_u^\top y_i)^2 + (1-\rho) \sum_{i \in \mathcal{L}} \text{CE}(y_i, g(\hat{r}_{\cdot i}))$$

2. **Key hyperparameters**: $\rho \in [0.3, 0.7]$, $d \in [10, 50]$, $\lambda \in [0.01, 0.1]$

3. **Algorithm**: Alternating Least Squares (ALS) for $X, Y$ + gradient descent for aggregator $g$

4. **Implementation**: Start with `src/cfensemble/optimization/` and `notebooks/`

---

## Questions?

For technical questions or implementation discussions, refer to:
- Implementation: `src/cfensemble/` source code
- Examples: `notebooks/` Jupyter notebooks
- Issues: GitHub issues (if available)
- Development notes: `dev/methods/` (private)

---

## Citation

If you use this work, please cite:

```bibtex
@article{cfensemble2024,
  title={CF-Ensemble: Knowledge Distillation Meets Collaborative Filtering for Ensemble Learning},
  author={Your Name},
  year={2024}
}
```

---

**Last Updated**: January 2026
