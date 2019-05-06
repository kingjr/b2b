import numpy as np
from scipy.sparse import eye
from scipy.linalg import pinv, svd
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA


class CCAJR(BaseEstimator):
    """
    """
    def __init__(self, n_components=-1, reg=0):
        self.reg = reg
        self.n_components = n_components

    def fit(self, X, Y):
        CYY = Y.T @ Y + self.reg * eye(Y.shape[1])
        CXX = X.T @ X + self.reg * eye(X.shape[1])
        CXY = X.T @ Y
        _U, _S, V = svd(pinv(CYY) @ CXY.T @ pinv(CXX) @ CXY)
        self.W_ = V[:self.n_components, :].T
        self.coef_ = pinv(CXX) @ CXY @ self.W_
        return self

    def predict(self, X):
        """Predict Y from X."""
        return X @ self.coef_ @ self.W_.T

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
    cca_jr = CCAJR(3)
    train, test = range(0, n, 2), range(1, n, 2)
    coef = cca_jr.fit(X[train], Y[train]).coef_
    E_hat = np.mean(coef**2, 1)
    score = cca_jr.score(X[test], Y[test])

    print('E_auc', roc_auc_score(np.diag(E), E_hat))
    print('Y_score', score)

    # compare with sklearn
    cca = CCA(3)
    train, test = range(0, n, 2), range(1, n, 2)
    coef = cca.fit(X[train], Y[train]).coef_
    E_hat = np.mean(coef**2, 1)
    score = cca.score(X[test], Y[test])

    print('sk: E_auc', roc_auc_score(np.diag(E), E_hat))
    print('sk: Y_score', score)
