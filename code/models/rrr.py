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


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = randn(num, dimX)
    W = randn(dimX, rrank)  @ randn(rrank, dimY)
    Y = X @ W + randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, rank, reg=0):
        self.reg = reg
        self.rank = rank

    def fit(self, X, Y):
        CXX = X.T @ X + self.reg * eye(X.shape[1])
        CXY = X.T @ Y
        _U, _S, V = svd(CXY.T @ pinv(CXX) @ CXY)
        self.W_ = V[:self.rank, :].T
        self.A_ = (pinv(CXX) @ CXY @ self.W_).T
        return self

    def predict(self, X):
        """Predict Y from X."""
        return X @ self.A_.T @ self.W_.T

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
    nY = 10
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
    coef = rrr.fit(X[train], Y[train]).A_
    E_hat = np.mean(coef**2, 0)
    score = rrr.score(X[test], Y[test])

    print('E_auc', roc_auc_score(np.diag(E), E_hat))
    print('Y_score', score)
