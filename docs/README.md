# Documentation

This directory contains official documentation for the CF-Ensemble project.

## Contents

- **[CF-EnsembleLearning-Intro.pdf](CF-EnsembleLearning-Intro.pdf)**: Introductory document on meta-learning via latent-factor-based collaborative filtering
- **[CF-based-ensemble-learning-slides.pdf](CF-based-ensemble-learning-slides.pdf)**: Presentation slides
- **[CFEnsembleLearning-optimization.pdf](CFEnsembleLearning-optimization.pdf)**: Detailed optimization formulation and methods

## Method Documentation

The `methods/` subdirectory contains detailed documentation on specific methods:

### 🎓 **NEW:** Essential Reading for Practitioners

- **[Imbalanced Data Tutorial](methods/imbalanced_data_tutorial.md)** 🆕 **START HERE**
  - Random baseline performance (PR-AUC, F1, ROC-AUC, Accuracy) for 1%, 5%, 10% minorities
  - Clinical significance: What performance is "good enough"?
  - State-of-the-art methods (2026): SMOTE, Focal Loss, Foundation Models, Active Learning
  - Where CF-Ensemble fits in: Optimal at 5-10% minority (+1-4% gains)
  - Complete code examples and decision trees

### Confidence Weighting (Evidence-Based, 2026)

- **[Confidence Weighting Documentation](methods/confidence_weighting/)** - Complete directory with:
  - **[When to Use Confidence Weighting](methods/confidence_weighting/when_to_use_confidence_weighting.md)** - Practitioner's guide ⭐
  - **[Base Classifier Quality Analysis](methods/confidence_weighting/base_classifier_quality_analysis.md)** - Detailed analysis
  - **[Theory vs. Empirics](methods/confidence_weighting/theory_vs_empirics.md)** - What can be proven?
  - **[Polarity Models Tutorial](methods/polarity_models_tutorial.md)** - Mathematical foundations

**Key validated findings:**
- 5-10% minority = optimal for CF-Ensemble (+1-4% gains) 🏆
- m ≥ 15 classifiers → simple averaging powerful (ensemble size effect)
- PR-AUC essential for imbalanced data (not ROC-AUC!)

### Optimization & Foundations

- **[cf_ensemble_optimization_objective.md](methods/cf_ensemble_optimization_objective.md)** - Optimization objectives for CF-based ensemble learning
- **[knowledge_distillation.md](methods/knowledge_distillation.md)** - Knowledge distillation approaches

## References

The `references/` subdirectory contains reference materials and papers.

## Development Notes

For development notes, temporary documentation, and learning materials, see the `dev/` directory at the project root.
