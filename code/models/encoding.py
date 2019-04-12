import numpy as np
from sklearn.linear_model import RidgeCV, MultiTaskLassoCV, LinearRegression
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score


class Encoding(BaseEstimator):
    def __init__(self, n_components=5, ols=None):
        self.pca = PCA(n_components)
        self.ols = RidgeCV() if ols is None else ols

    def fit(self, X, Y):
        Yt = self.pca.fit_transform(Y)
        self.ols.fit(X, Yt)
        self.coef_ = self.ols.coef_.T  # to be in similar format to PLS and CCA
        return self

    def predict(self, X):
        Ypred = self.ols.predict(X)
        return self.pca.inverse_transform(Ypred)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
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
    alphas = np.logspace(-2, 2, 5)
    penalties = dict(none=LinearRegression(),
                     l2=RidgeCV(alphas=alphas),
                     l1=MultiTaskLassoCV(alphas=alphas))
    for penalty, model in penalties.items():
        encod = Encoding(ols=model)
        train, test = range(0, n, 2), range(1, n, 2)
        coef = encod.fit(X[train], Y[train]).coef_
        E_hat = np.mean(coef**2, 1)
        score = encod.score(X[test], Y[test])

        print('%s: E_auc=%.2f; Y_score=%.2f' % (
            penalty, roc_auc_score(np.diag(E), E_hat), score))
