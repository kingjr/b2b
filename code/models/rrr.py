"""
Reduced rank regression class.
aka Optimal linear 'bottlenecking' or 'multitask learning'.

Aapted from
https://raw.githubusercontent.com/riscy/machine_learning_linear_models/master/reduced_rank_regressor.py

(c) Chris Rayner (2015), dchrisrayner AT gmail DOT com

For kernel implementation see
https://github.com/rockNroll87q/RRRR/blob/master/reduced_rank_regressor.py
"""


import numpy as np
from scipy import randn
from scipy.sparse import eye
from scipy.linalg import pinv, svd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = randn(num, dimX)
    W = randn(dimX, rrank)  @ randn(rrank, dimY)
    Y = X @ W + randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(BaseEstimator):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, n_components=-1, reg=0):
        self.reg = reg
        self.n_components = n_components

    def fit(self, X, Y):
        CXX = X.T @ X + self.reg * eye(X.shape[1])
        CXY = X.T @ Y
        _U, _S, V = svd(CXY.T @ pinv(CXX) @ CXY)
        if self.n_components != -1:
            V = V[:self.n_components, :]
        self.W_ = V.T
        self.coef_ = pinv(CXX) @ CXY @ self.W_
        return self

    def predict(self, X):
        """Predict Y from X."""
        return X @ self.coef_ @ self.W_.T

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


class RRR(BaseEstimator):
    """JR implementation of regularized reduced rank regression"""
    def __init__(self, n_components=-1, alpha=0.):
        self.alpha = alpha
        self.n_components = n_components

    def fit(self, X, Y):
        # Regularized inv(X'X)
        if isinstance(self.alpha, (float, int)):
            if self.alpha == 0:
                least_square = LinearRegression(fit_intercept=False)
            else:
                least_square = Ridge(alpha=self.alpha, fit_intercept=False)
        else:
            least_square = RidgeCV(alphas=self.alpha, fit_intercept=False)
        coef = least_square.fit(X, Y).coef_.T

        Y_hat = least_square.predict(X)
        _U, _S, V = svd(Y.T @ Y_hat)
        if self.n_components != -1:
            V = V[:self.n_components, :]
        self.coef_ = coef @ V.T @ V
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
    from sklearn.preprocessing import scale
    from sklearn.metrics import roc_auc_score
    # Simulate data
    """Y = F(EX+N)"""

    np.random.seed(0)

    # Problem dimensionality
    n = 1000
    nE = nX = 10
    nY = 20
    snr = .25  # signal to noise ratio
    selected = .5  # number of X feature selected by E

    selected = min(int(np.floor(selected*nX)) + 1, nX-1)
    E = np.identity(nX)
    E[selected:] = 0

    # X covariance
    Cx = np.random.randn(nX, nX)
    Cx = Cx.dot(Cx.T) / nX  # sym pos-semidefin
    X = np.random.multivariate_normal(np.zeros(nX), Cx, n)

    # Noise (homosedastic in source space)
    N = np.random.randn(n, nE)

    # Forward operator (linear mixture)
    F = np.random.randn(nY, nE)

    Y = ((X @ E.T) * snr + N) @ F.T

    X = scale(X)
    Y = scale(Y)

    # Fit method
    rrr = ReducedRankRegressor(3)
    train, test = range(0, n, 2), range(1, n, 2)
    coef = rrr.fit(X[train], Y[train]).coef_
    E_hat = np.mean(coef**2, 1)
    score = rrr.score(X[test], Y[test])

    print('E_auc', roc_auc_score(np.diag(E), E_hat))
    print('Y_score', score)

    # jr implementation
    rrr = RRR(3)
    train, test = range(0, n, 2), range(1, n, 2)
    coef = rrr.fit(X[train], Y[train]).coef_
    E_hat = np.mean(coef**2, 1)
    score = rrr.score(X[test], Y[test])

    print('jr: E_auc', roc_auc_score(np.diag(E), E_hat))
    print('jr: Y_score', score)
