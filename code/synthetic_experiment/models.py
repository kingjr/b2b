from utils import sonquist_morgan
from sklearn.linear_model import RidgeCV, LinearRegression
import numpy as np


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


class OLS(object):
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.coef = basic_regression(X, Y)
        return self

    def predict(self, X):
        return X @ self.coef

    def solution(self):
        return sonquist_morgan(np.power(self.coef, 2).sum(1))


class Oracle(object):
    def __init__(self, true_mask):
        self.true_mask = np.diag(true_mask)

    def fit(self, X, Y):
        self.coef = basic_regression(X @ self.true_mask, Y)

    def predict(self, X):
        return X @ self.true_mask @ self.coef

    def solution(self):
        return np.diag(self.true_mask)
