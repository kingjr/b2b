"""David's first approach when I exposed the problem.
Reasonable to add in the comparison?
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit


def correlation(x, y):
    a = (x - x.mean(0)) / x.std(0)
    b = (y - y.mean(0)) / y.std(0)
    return a.T @ b / x.shape[0]


def partial_correlation_bagging(solver, x, y, z, ensemble=None):
    if ensemble is None:
        ensemble = [(range(len(x)), range(len(x))), ]
    r = []
    for set1, set2 in ensemble:
        p_x = solver.fit(z[set1], x[set1]).predict(z[set2])
        p_y = solver.fit(z[set1], y[set1]).predict(z[set2])
        r.append(correlation(x[set2] - p_x, y[set2] - p_y))
    return np.mean(r, 0)


def partial_correlation_loop(solver, x, y, ensemble=None):
    e_hat = np.zeros(y.shape[1])
    for i in range(y.shape[1]):
        y_i = y[:, i].reshape(-1, 1)
        y_not_i = np.delete(y, i, axis=1)
        r = partial_correlation_bagging(solver, x, y_i, y_not_i, ensemble)
        e_hat[i] = np.sum(r**2)
    return e_hat


class PartialCorrelation(object):

    def __init__(self, solver=None, bagging=False):
        self.solver = RidgeCV() if solver is None else solver
        self.bagging = bagging

    def fit(self, X, Y):
        ensemble = None
        if self.bagging:
            cv = ShuffleSplit(test_size=.5)
            ensemble = [(train, test) for train, test in cv.split(X, Y)]
        self.E_ = partial_correlation_loop(self.solver, X, Y, ensemble)
        return self


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
    snr = 25  # signal to noise ratio
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
    partialcorr = PartialCorrelation()
    train, test = range(0, n, 2), range(1, n, 2)
    E_hat = partialcorr.fit(X[train], Y[train]).E_
    # score = partialcorr.score(X[test], Y[test])  # TODO

    print('E_auc', roc_auc_score(np.diag(E), E_hat))
