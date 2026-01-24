import numpy as np
# import common

import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.style.use('seaborn')

from utils_plot import saveFig
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

"""

Reference 
---------
1. sklearn-crfsuite
   pip install sklearn-crfsuite
   https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
"""

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian generative classification based on KDE.
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        """
        predict_proba() returns an array of class probabilities of shape [n_samples, n_classes]

        Entry [i, j] of this array is the posterior probability that sample i is a member of class j, 
        computed by multiplying the likelihood by the class prior and normalizing.
        """
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


class CESClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self): 
        pass 

# Utility function to report best scores
def report(results, n_top=3):
    """
    Params
    ------
    results: am output dictionary from a GridSearchCV or RandomizedSearchCV
             as in grid_search.cv_results_ (which is a dictionary)

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def optimize_crf_params(X, y, model, labels, max_size=3000, verfiy=True): 
    import scipy.stats as stats
    from sklearn.metrics import make_scorer
    # import sklearn_crfsuite
    from sklearn_crfsuite import scorers
    from sklearn_crfsuite import metrics

    params_space = {
        'c1': stats.expon(scale=0.5),
        'c2': stats.expon(scale=0.05),
    }

    # use f1 for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score, 
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(model, params_space, 
                            cv=3, 
                            verbose=1, 
                            n_jobs=-1, 
                            n_iter=50, 
                            scoring=f1_scorer)

    X_train, y_train = X, y 
    N = len(X_train)
    if N > max_size: 
        indices = np.random.choice(range(N), max_size)
        X_train = list( np.asarray(X_train)[indices] )
        y_train = list( np.asarray(y_train)[indices] )

    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    # if verify: validate_crf_params(rs, output_path=None, dpi=300)

    return rs.best_estimator_

def validate_crf_params(rs, output_path=None, dpi=300, save=True, verbose=True): 
    _x = [s.parameters['c1'] for s in rs.cv_results_]
    _y = [s.parameters['c2'] for s in rs.cv_results_]
    _c = [s.mean_validation_score for s in rs.cv_results_]

    plt.clf() 

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

    print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    if save: 
        basedir = os.path.join(os.getcwd(), 'analysis')
        if not os.path.exists(basedir) and create_dir:
            print('(validate_crf_params) Creating analysis directory:\n%s\n' % basedir)
            os.mkdir(basedir) 

        if output_path is None: 
            if not name: name = 'generic'
            fname = '{prefix}.P-{suffix}-{index}.{ext}'.format(prefix='{}-crf'.format(kernel), suffix=name, index=index, ext=ext)
            output_path = os.path.join(basedir, fname)  # example path: System.analysisPath
        else: 
            # output_path can be either a file name OR a full path including the file name
            prefix, fname = os.path.dirname(output_path), os.path.basename(output_path)
            if not prefix: 
                prefix = basedir
                output_path = os.path.join(basedir, fname)
            assert os.path.exists(output_path), "Invalid output path: {}".format(output_path)

        if verbose: print('(validate_crf_params) Saving distribution plot at: {path}'.format(path=output_path))
        saveFig(plt, output_path, dpi=dpi, verbose=verbose) 
    else: 
        # pass
        try: 
            plt.show()
        except: 
            pass
    return
    

def fmax_score_threshold(labels, predictions, beta = 1.0, pos_label = 1):
    """

    Memo
    ---- 
    1. precision and recall tradeoff doesn't take into account true negative
       
       precision: Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1. 
       recall: Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.

    2. example 

    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision  
    array([0.66666667, 0.5       , 1.        , 1.        ])
    >>> recall
    array([1. , 0.5, 0.5, 0. ])
    >>> thresholds
    array([0.35, 0.4 , 0.8 ])

    precision[1] = 0.5, for any prediction >= thresholds[1] = 0.4 as positive (assuming that pos_label = 1)

    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(labels, predictions, pos_label)

    # the general formula for positive beta
    # ... if beta == 1, then this is just f1 score, harmonic mean between precision and recall 
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    i = np.nanargmax(f1)  # the position for which f1 is the max 
    th = threshold[i] if i < len(threshold) else 1.0    # len(threshold) == len(precision) -1 
    # assert f1[i] == nanmax(f1)
    return (f1[i], th)

def load_iris(): 
    """

    Memo
    ----
    1. n_samples: 100, n_features: 804, n_classes: 2
    """
    from sklearn import datasets
    # Data IO and generation

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
    print('(data) n_samples: {n}, n_features: {nf}'.format(n=n_samples, nf=n_features))

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    return (X, y)

def t_kde_classifier(): 
    from sklearn.datasets import load_digits
    from sklearn.grid_search import GridSearchCV

    digits = load_digits()

    bandwidths = 10 ** np.linspace(0, 2, 100)
    grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
    grid.fit(digits.data, digits.target)

    scores = [val.mean_validation_score for val in grid.grid_scores_]

    plt.semilogx(bandwidths, scores)
    plt.xlabel('bandwidth')
    plt.ylabel('accuracy')
    plt.title('KDE Model Performance')
    print(grid.best_params_)
    print('accuracy =', grid.best_score_)

    return

def test(**kargs): 
    from sklearn.model_selection import train_test_split

    estimator = MeanClassifier()
    X, y = load_iris()
    print('> dim(X): {0} | dim(y): {1}'.format(X.shape, y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    print('> dim(X_train): {0} | dim(y_train): {1}'.format(X_train.shape, y_train.shape))
    estimator.fit(X_train, y_train)
    print('> test example: {0}'.format(X_test[:10]))

    y_score = estimator.predict_proba(X_test, y_test)
    print('> scores:       {0}'.format(y_score[:10]))

    return

if __name__ == "__main__":
    test() 



