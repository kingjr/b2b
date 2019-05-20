from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
import numpy as np

N_JOBS = 1

def sonquist_morgan(x):
    z = np.sort(x)
    n = z.size
    m1 = 0
    m2 = np.sum(z)
    mx = 0
    best = -1
    for i in range(n - 1):
        m1 += z[i]
        m2 -= z[i]
        ind = (i + 1) * (n - i - 1) * (m1 / (i + 1) - m2 / (n - i - 1))**2
        if ind > mx:
            mx = ind
            best = z[i]
    res = [0 for i in range(n)]
    for i in range(n):
        if x[i] > best:
            res[i] = 1
    return np.array(res)


def basic_regression(X, Y, regularize=True):
    if regularize:
        alphas = np.logspace(-5, 5, 20)
        w = RidgeCV(fit_intercept=False, alphas=alphas).fit(X, Y).coef_
    else:
        w = LinearRegression(fit_intercept=False).fit(X, Y).coef_

    return w.T


class JRR(object):
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    def fit(self, X, Y):
        E = 0
        for _ in range(self.n_splits):
            perm = np.random.permutation(range(len(X)))
            G = basic_regression(Y[perm[0::2]], X[perm[0::2]])
            E += np.diag(basic_regression(X[perm[1::2]], Y[perm[1::2]] @ G))

        self.E = sonquist_morgan(E / self.n_splits)
        self.coef = basic_regression(X @ np.diag(self.E), Y)

    def predict(self, X):
        return X @ np.diag(self.E) @ self.coef

    def solution(self):
        return self.E


class Oracle(object):
    def __init__(self, true_mask):
        self.true_mask = np.diag(true_mask)

    def fit(self, X, Y):
        self.coef = basic_regression(X @ self.true_mask, Y)

    def predict(self, X):
        return X @ self.true_mask @ self.coef

    def solution(self):
        return np.diag(self.true_mask)


class SKModel(BaseEstimator):
    def __init__(self):
        pass

    def predict(self, X):
        return X @ self.coef

    def solution(self):
        return sonquist_morgan(np.power(self.coef, 2).sum(1))


class OLS(SKModel):
    def fit(self, X, Y):
        self.coef = LinearRegression(fit_intercept=False).fit(X, Y).coef_.T
        return self


class Ridge(SKModel):
    def fit(self, X, Y):
        self.coef = RidgeCV(fit_intercept=False).fit(X, Y).coef_.T
        return self


class PLS(SKModel):
    def fit(self, X, Y):
        grid = {"n_components": np.linspace(1, X.shape[1], 10).astype(int)}
        pls = GridSearchCV(PLSRegression(), grid, n_jobs=N_JOBS, cv=5)
        self.coef = pls.fit(X, Y).best_estimator_.coef_
        return self


class Lasso(SKModel):
    def fit(self, X, Y, max_samples=100):
        p = np.random.permutation(len(X))
        if len(p) > max_samples:
            p = p[:max_samples]

        self.coef = MultiTaskLassoCV(fit_intercept=False,
                                     selection="random",
                                     n_alphas=10,
                                     n_jobs=N_JOBS,
                                     cv=5).fit(X[p], Y[p]).coef_.T
        return self


class RRROne(SKModel):
    """
    Adapted from: https://github.com/riscy/machine_learning_linear_models
    """

    def __init__(self, n_components=-1, reg=0):
        self.reg = reg
        self.n_components = n_components

    def fit(self, X, Y):
        CXX = X.T @ X + self.reg * np.eye(X.shape[1])
        CXY = X.T @ Y
        _U, _S, V = np.linalg.svd(CXY.T @ np.linalg.pinv(CXX) @ CXY)
        if self.n_components != -1:
            V = V[:self.n_components, :]
        W = V.T
        A = np.linalg.pinv(CXX) @ CXY @ W
        self.coef = A @ W.T
        return self

    def score(self, X, y):
        return r2_score(y, self.predict(X), multioutput='variance_weighted')


class RRR(SKModel):
    def fit(self, X, Y):
        grid = {
            "n_components": np.linspace(1, X.shape[1], 10).astype(int),
            "reg": np.logspace(-3, 3, 10)
        }

        rrr = GridSearchCV(RRROne(), grid, n_jobs=N_JOBS, cv=5)
        self.coef = rrr.fit(X, Y).best_estimator_.coef
        return self
