"""

https://en.wikipedia.org/wiki/Canonical_correlation

See github.com/Neuromorphs/telluride-decoding-toolbox/blob/master/cca.m
for similar approach without scaling and bagging
"""

import numpy as np
from scipy.linalg import eigh, pinv
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_decomposition import CCA as skCCA


class CCA(BaseEstimator):

    def __init__(self, n_components=-1, alpha=0., normalize=True,
                 bagging=False):
        self.n_components = n_components
        self.alpha = alpha
        self.bagging = bagging
        self.normalize = normalize

    def fit(self, X, Y):
        assert len(X) == len(Y)

        nx = X.shape[1]
        ny = Y.shape[1]

        if self.normalize:
            self.xscale_ = StandardScaler()
            X = self.xscale_.fit_transform(X)
            self.yscale_ = StandardScaler()
            Y = self.yscale_.fit_transform(Y)

        if self.bagging in (False, 0, None):
            Gset = range(len(X))
            Hset = range(len(X))
            ensemble = [(Gset, Hset)]
        else:
            bagging = ShuffleSplit(self.bagging, test_size=.5)
            ensemble = [split for split in bagging.split(X, Y)]

        nx, ny = X.shape[1], Y.shape[1]

        # Bagging ensemble
        Pxxs = list()
        Cxys = list()
        Pyys = list()
        Cyxs = list()
        for set1, set2 in ensemble:
            # FIXME: suboptimal way to compute cov(X,Y)
            XY1 = np.c_[X[set1], Y[set1]]
            XY2 = np.c_[X[set2], Y[set2]]
            Cxy = np.cov(XY1.T)[:nx, nx:]
            Cyx = np.cov(XY2.T)[:nx, nx:].T

            Cxx = np.cov(X[set1].T)
            Cyy = np.cov(Y[set2].T)

            # invert matrices with tikohnov regularization
            Pxxs.append(pinv(Cxx + self.alpha*np.eye(nx)))
            Pyys.append(pinv(Cyy + self.alpha*np.eye(ny)))
            Cxys.append(Cxy)
            Cyxs.append(Cyx)

        # mean across bags
        Pxx = np.mean(Pxxs, 0)
        Cxy = np.mean(Cxys, 0)
        Pyy = np.mean(Pyys, 0)
        Cyx = np.mean(Cyxs, 0)

        eigvals, Wx = eigh(Pxx @ Cxy @ Pyy @ Cyx)

        # order eigen values
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        Wx = Wx[:, order]

        # truncate Z space
        n = max(self.n_components, nx)
        Wx = Wx[:n]
        eigvals = eigvals

        # compute Z to Y transform
        Wy = Pyy @ Cyx @ Wx
        Wy /= (Wy**2).sum(0)
        Py = pinv(Wy.T)  # FIXME: inverse non square matrix?

        Z = np.diag(eigvals)
        self.coef_ = Wx @ Z @ Py.T
        self.Wx_ = Wx
        self.Wy_ = Wy
        self.Py_ = Py
        self.eigvals_ = eigvals

        return self

    def predict(self, X):
        if self.normalize:
            X = self.xscale_.transform(X)

        Y_hat = X @ self.coef_

        if self.normalize:
            Y_hat = self.yscale_.inverse_transform(Y_hat)
        return Y_hat

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
    # Initialize number of samples
    n_samples = 1000

    # Define two latent variables (number of samples x 1)
    L1, L2 = np.random.randn(2, n_samples,)

    # Define independent components for each dataset
    # (number of observations x dataset dimensions)
    indep1 = np.random.randn(n_samples, 4)
    indep2 = np.random.randn(n_samples, 5)

    # Create two datasets, with each dimension composed as
    # a sum of 75% one of the latent variables and 25% independent component
    X = .25*indep1 + .75*np.vstack((L1, L2, L1, L2)).T
    Y = .25*indep2 + .75*np.vstack((L1, L2, L1, L2, L1)).T

    # Split each dataset into two halves: training set and test set
    train = range(0, len(X), 2)
    test = range(1, len(X), 2)

    cca = CCA(n_components=2, alpha=0, bagging=None)
    print(cca.fit(X[train], Y[train]).score(X[test], Y[test]))

    # FIXME: my implementation is clearly unstable

    skcca = skCCA(n_components=2)
    print(skcca.fit(X[train], Y[train]).score(X[test], Y[test]))
