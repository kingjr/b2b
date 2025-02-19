from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from itertools import product
import scipy as sc
import numpy as np


N_JOBS = 1


def ridge_cv(X, Y, alphas, independent=False):
    """
    Similar to sklearn RidgeCV but
    (1) can optimize a different alpha for each column of Y
    (2) return leave-one-out Y_hat
    """
    if isinstance(alphas, (float, int)):
        alphas = np.array([alphas, ], np.float64)
    alphas = np.asarray(alphas)
    n, n_x = X.shape
    n, n_y = Y.shape
    # Decompose X
    U, s, _ = np.linalg.svd(X, full_matrices=0)
    v = s**2
    UY = U.T @ Y

    # For each alpha, solve leave-one-out error coefs
    cv_duals = np.zeros((len(alphas), n, n_y))
    cv_errors = np.zeros((len(alphas), n, n_y))
    for alpha_idx, alpha in enumerate(alphas):
        # Solve
        w = ((v + alpha) ** -1) - alpha ** -1
        c = U @ np.diag(w) @ UY + alpha ** -1 * Y
        cv_duals[alpha_idx] = c

        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        G_diag = (w * U ** 2).sum(axis=-1) + alpha ** -1
        error = c / G_diag[:, np.newaxis]
        cv_errors[alpha_idx] = error

    # identify best alpha for each column of Y independently
    if independent:
        best_alphas = (cv_errors ** 2).mean(axis=1).argmin(axis=0)
        duals = np.transpose([cv_duals[b, :, i]
                              for i, b in enumerate(best_alphas)])
        cv_errors = np.transpose([cv_errors[b, :, i]
                                  for i, b in enumerate(best_alphas)])
    else:
        _cv_errors = cv_errors.reshape(len(alphas), -1)
        best_alphas = (_cv_errors ** 2).mean(axis=1).argmin(axis=0)
        duals = cv_duals[best_alphas]
        cv_errors = cv_errors[best_alphas]

    coefs = duals.T @ X
    Y_hat = Y - cv_errors
    return coefs.T, Y_hat


class MyRidgeCV(object):
    def __init__(self, alphas=np.logspace(-5, 5, 20), independent=True):
        self.alphas = alphas
        self.independent = independent

    def __call__(self, X, Y):
        return ridge_cv(X, Y, self.alphas, self.independent)


def basic_regression(X, Y, regularize=True):
    if regularize:
        alphas = np.logspace(-5, 5, 20)
        w = RidgeCV(fit_intercept=False, alphas=alphas).fit(X, Y).coef_
    else:
        w = LinearRegression(fit_intercept=False).fit(X, Y).coef_

    return w.T


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
        return np.power(self.coef, 2).sum(1)


class OLS(SKModel):
    def fit(self, X, Y):
        self.coef = LinearRegression(fit_intercept=False).fit(X, Y).coef_.T
        return self


class Ridge(SKModel):
    def fit(self, X, Y):
        self.coef = RidgeCV(fit_intercept=False,
                            alphas=np.logspace(-5, 5, 10)).fit(X, Y).coef_.T
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
                                     max_iter=10000,
                                     n_alphas=10,
                                     n_jobs=N_JOBS,
                                     cv=5).fit(X[p], Y[p]).coef_.T
        return self


class ReducedRankRegression(SKModel):
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
            "reg": np.logspace(-5, 5, 10)
        }

        rrr = GridSearchCV(ReducedRankRegression(), grid, n_jobs=N_JOBS, cv=5)
        self.coef = rrr.fit(X, Y).best_estimator_.coef
        return self


class CanonicalCorrelation(SKModel):
    def __init__(self, n_components=-1, reg=0):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, Y):
        if self.n_components < 1:
            self.n_components = min(X.shape[1], Y.shape[1])

        CXX = (X.T @ X) / X.shape[0]
        CYY = (Y.T @ Y) / Y.shape[0]
        CXY = (X.T @ Y) / X.shape[0]

        sqrtCX = sc.linalg.sqrtm(CXX + np.eye(CXX.shape[0]) * self.reg)
        sqrtCY = sc.linalg.sqrtm(CYY + np.eye(CYY.shape[0]) * self.reg)

        isqrtCX = sc.linalg.inv(sqrtCX)
        isqrtCY = sc.linalg.inv(sqrtCY)

        U, S, V = np.linalg.svd(isqrtCX @ CXY @ isqrtCY)

        A = isqrtCX @ U[:, :self.n_components]
        B = isqrtCY @ V.T[:, :self.n_components]
        self.coef = A @ A.T @ X.T @ Y / X.shape[0]

    def predict(self, X):
        return X @ self.coef

    def score(self, X, y):
        return r2_score(y, self.predict(X), multioutput='variance_weighted')


class CCA(SKModel):
    def fit(self, X, Y):
        m = min(X.shape[1], Y.shape[1])
        grid = {
            "n_components": np.linspace(1, m, 10).astype(int),
            "reg": np.logspace(-5, 5, 10)
        }

        cca = GridSearchCV(CanonicalCorrelation(), grid, n_jobs=N_JOBS, cv=5)
        self.coef = cca.fit(X, Y).best_estimator_.coef
        return self


class JRR(object):
    def __init__(self):
        alphas = np.logspace(-5, 5, 20)
        self.G = MyRidgeCV(alphas, independent=True)
        self.H = MyRidgeCV(alphas, independent=False)
        self.n_splits = 100

    def fit(self, X, Y):
        E = 0
        for _ in range(self.n_splits):
            perm = np.random.permutation(range(len(X)))
            G, _ = self.G(Y[perm[0::2]], X[perm[0::2]])
            H, _ = self.H(X[perm[1::2]], Y[perm[1::2]] @ G)
            E += np.diag(H)

        self.E = E / self.n_splits

        self.coef = basic_regression(X @ np.diag(self.E), Y)

    def predict(self, X):
        return X @ np.diag(self.E) @ self.coef

    def solution(self):
        return self.E
