import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class EstimatorCV(BaseEstimator):
    """Transform an operator into a grid search for n_componentsself.
    This is useful because sklearn CCA and PLS restricts the n_components
    to the X dimensions whereas our Encoding model resticts the n_components to
    the Y.
    """

    def __init__(self, estimator, dim='X', n_components=5):
        """
        n_components is the number of models tested between 1 and size of 'dim'
        e.g. if dim == 'X' and X.shape[1] == 100 and n_components = 3
             then the grid will search through self.components_= [1, 10, 100]
        """
        self.n_components = n_components
        self.dim = dim
        self.base_estimator = estimator

    def fit(self, X, y):
        if self.dim == 'X':
            n_max = X.shape[1]
        else:
            n_max = y.shape[1]

        components = np.logspace(0, np.log10(n_max), self.n_components)
        components = np.unique(components.astype(int))
        self.estimator_ = GridSearchCV(self.base_estimator,
                                       dict(n_components=components))
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)


if __name__ == '__main__':
    from sklearn.cross_decomposition import CCA
    cca_cv = EstimatorCV(CCA(), dim='X')
