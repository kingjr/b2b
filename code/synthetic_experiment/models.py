from sklearn.linear_model import RidgeCV, LinearRegression
import numpy as np

def sonquist_morgan(x):
    z=np.sort(x)
    n=z.size
    m1=0
    m2=np.sum(z)
    mx=0
    best=-1
    for i in range(n-1):
        m1+=z[i]
        m2-=z[i]
        ind=(i+1)*(n-i-1)*(m1/(i+1)-m2/(n-i-1))**2
        if ind>mx :
            mx=ind
            best=z[i]
    res=[0 for i in range(n)]
    for i in range(n):
        if x[i]>best: res[i] = 1
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
