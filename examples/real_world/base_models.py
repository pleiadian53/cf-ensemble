"""
Heterogeneous base classifiers for building the probability matrix R.

Mirrors the spirit of the original CF-ensemble experiments (12 diverse WEKA
classifiers): probabilistic, linear, instance-based, rule/tree-based, boosted and
neural models, so their errors are complementary rather than redundant. Defaults
are deliberately untuned and imbalance-agnostic — realistic, miscalibrated
"teachers" are what CF-Ensemble is meant to combine.
"""

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def make_base_models(seed: int = 0, n_jobs: int = -1) -> Dict[str, BaseEstimator]:
    """Return {name: unfitted estimator}; every estimator supports predict_proba."""

    def scaled(est):
        return make_pipeline(StandardScaler(), est)

    return {
        "gaussian_nb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "logreg": scaled(LogisticRegression(max_iter=2000)),
        "linear_svm": scaled(CalibratedClassifierCV(LinearSVC(C=0.1), method="sigmoid", cv=3)),
        "knn": scaled(KNeighborsClassifier(n_neighbors=25, n_jobs=n_jobs)),
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed
        ),
        "hist_gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed),
        "adaboost": AdaBoostClassifier(n_estimators=200, random_state=seed),
        "mlp": scaled(MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=seed
        )),
    }
