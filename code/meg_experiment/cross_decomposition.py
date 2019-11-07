import numpy as np
from scipy import linalg
from sklearn.cross_decomposition import CCA as SkCCA
from sklearn.cross_decomposition import PLSRegression as SkPLS
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from base import rn_score


def canonical_correlation(model, X, Y, scoring, multioutput):
    """Score in canonical space"""

    # check valid model
    for xy in 'xy':
        for var in ('mean', 'std', 'rotations'):
            assert hasattr(model, '%s_%s_' % (xy, var))
    assert model.x_rotations_.shape[1] == model.y_rotations_.shape[1]

    # check valid data
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim == 1:
        X = X[:, None]
    assert len(X) == len(Y)

    # Project to canonical space
    X = X - model.x_mean_
    X /= model.x_std_
    X = np.nan_to_num(X, 0)
    XA = X @ model.x_rotations_

    Y = Y - model.y_mean_
    Y /= model.y_std_
    Y = np.nan_to_num(Y, 0)
    YB = Y @ model.y_rotations_

    return rn_score(XA, YB, scoring=scoring, multioutput=multioutput)


class CCA(SkCCA):
    """overwrite scikit-learn CCA to get propper scoring function"""

    def __init__(self, n_components=2,
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.scoring = scoring
        self.multioutput = multioutput
        super().__init__(n_components=n_components, tol=tol)

    def fit(self, X, Y):
        super().fit(X, Y)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)


class PLS(SkPLS):
    """overwrite scikit-learn PLSRegression to get propper scoring function"""

    def __init__(self, n_components=2,
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.scoring = scoring
        self.multioutput = multioutput
        super().__init__(n_components=n_components, tol=tol)

    def fit(self, X, Y):
        super().fit(X, Y)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)


class RegCCA(CCA):
    """Wrapper to get sklearn API for Regularized CCA """
    def __init__(self, alpha=0., n_components=-1,
                 scoring='r', multioutput='uniform_average',
                 tol=1e-15):
        self.alpha = alpha
        self.n_components = n_components
        assert (n_components > 0) or (n_components == -1)
        self.tol = tol
        self.scoring = scoring
        self.multioutput = multioutput

    def fit(self, X, Y):
        # Set truncation
        dx, dy = X.shape[1], Y.shape[1]
        dz_max = min(dx, dy)
        dz = dz_max if self.n_components == -1 else self.n_components
        dz = min(dz, dz_max)
        self.n_components_ = dz
        self.x_rotations_ = np.zeros((dx, dz))
        self.y_rotations_ = np.zeros((dy, dz))

        # Preprocess
        self.x_std_ = X.std(0)
        self.y_std_ = Y.std(0)
        self.x_mean_ = X.mean(0)
        self.y_mean_ = Y.mean(0)
        self.x_valid_ = self.x_std_ > 0
        self.y_valid_ = self.y_std_ > 0
        X = (X - self.x_mean_) / self.x_std_
        Y = (Y - self.y_mean_) / self.y_std_

        # compute cca
        comps = self._compute_kcca([X[:, self.x_valid_],
                                    Y[:, self.y_valid_]],
                                   reg=self.alpha, numCC=dz)
        self.x_rotations_[self.x_valid_] = comps[0]
        self.y_rotations_[self.y_valid_] = comps[1]

        return self

    def _compute_kcca(self, data, reg=0., numCC=None):
        """Adapted from https://github.com/gallantlab/pyrcca

        Copyright (c) 2015, The Regents of the University of California
        (Regents). All rights reserved.

        Permission to use, copy, modify, and distribute this software and its
        documentation for educational, research, and not-for-profit purposes,
        without fee and without a signed licensing agreement, is hereby
        granted, provided that the above copyright notice, this paragraph and
        the following two paragraphs appear in all copies, modifications, and
        distributions. Contact The Office of Technology Licensing, UC Berkeley,
        2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510)
        643-7201, for commercial licensing opportunities.

        Created by Natalia Bilenko, University of California, Berkeley.
        """

        kernel = [d.T for d in data]

        nDs = len(kernel)
        nFs = [k.shape[0] for k in kernel]
        numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

        # Get the auto- and cross-covariance matrices
        crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

        # Allocate left-hand side (LH) and right-hand side (RH):
        LH = np.zeros((sum(nFs), sum(nFs)))
        RH = np.zeros((sum(nFs), sum(nFs)))

        # Fill the left and right sides of the eigenvalue problem
        for i in range(nDs):
            RH[sum(nFs[:i]): sum(nFs[:i+1]),
               sum(nFs[:i]): sum(nFs[:i+1])] = (crosscovs[i * (nDs + 1)]
                                                + reg * np.eye(nFs[i]))

            for j in range(nDs):
                if i != j:
                    LH[sum(nFs[:j]): sum(nFs[:j+1]),
                       sum(nFs[:i]): sum(nFs[:i+1])] = crosscovs[nDs * j + i]

        LH = (LH + LH.T) / 2.
        RH = (RH + RH.T) / 2.

        maxCC = LH.shape[0]
        r, Vs = linalg.eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
        r[np.isnan(r)] = 0
        rindex = np.argsort(r)[::-1]
        comp = []
        Vs = Vs[:, rindex]
        for i in range(nDs):
            comp.append(Vs[sum(nFs[:i]):sum(nFs[:i + 1]), :numCC])
        return comp


class CrossKnockout(object):
    """Parent class to fit knockout cross decomposition model"""
    def __init__(self, model):
        self.model = model

        m = model.estimator if isinstance(model, GridSearchCV) else model
        assert isinstance(m, (CCA, PLS, RegCCA))

    def fit(self, X, Y):
        self.model.fit(X, Y)

        if isinstance(self.model, GridSearchCV):
            self.model_ = self.model.best_estimator_
        else:
            self.model_ = self.model

        # fit_knockout
        n_x = X.shape[1]
        self.knockout_models_ = list()
        for xi in np.arange(n_x):

            knockout = np.diag(np.ones(n_x))
            knockout[xi] = 0
            ko_model = clone(self.model_).fit(X @ knockout, Y)
            self.knockout_models_.append(ko_model)

        self.E_ = np.sum(self.model_.x_rotations_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        return self.model_.score(X, Y, scoring, multioutput)

    def score_knockout(self, X, Y, scoring=None, multioutput=None):
        n_x = X.shape[1]

        K_scores = list()
        for xi, model in enumerate(self.knockout_models_):

            knockout = np.diag(np.ones(n_x))
            knockout[xi] = 0

            R = model.score(X @ knockout, Y, scoring, multioutput)
            K_scores.append(R)

        return np.array(K_scores)


if __name__ == '__main__':

    def make_data():

        # Initialize number of samples
        n_samples = 1000

        # Define two latent variables (number of samples x 1)
        L1, L2 = np.random.randn(2, n_samples,)

        # Define independent components for each dataset
        # (number of observations x dataset dimensions)
        indep1 = np.random.randn(n_samples, 4)
        indep2 = np.random.randn(n_samples, 5)

        # Create two datasets, with each dimension composed as
        # a sum of 75% one of the latent variables and 25% independent comp
        X = .25*indep1 + .75*np.vstack((L1, L2, L1, L2)).T
        Y = .25*indep2 + .75*np.vstack((L1, L2, L1, L2, L1)).T

        # Split each dataset into two halves: training set and test set
        train = range(0, len(X), 2)
        test = range(1, len(X), 2)

        return X, Y, train, test

    X, Y, train, test = make_data()

    for scoring in ('r', 'r2'):
        for mo in ('variance_weighted', 'uniform_average'):
            params = dict(scoring=scoring, multioutput=mo)
            models = dict(
                cca2=CCA(2, **params), cca4=CCA(4, **params),
                pls2=PLS(2, **params), pls4=PLS(4, **params),
                regcca2=RegCCA(.001, n_components=2, **params),
                regcca4=RegCCA(.001, n_components=4, **params),
                regcca4_alpha1000=RegCCA(100., n_components=4, **params),
                regcca2_alpha100=RegCCA(100., n_components=2, **params))

            for name, model in models.items():
                model.fit(X[train], Y[train])
                print(scoring, mo, name, model.score(X[test], Y[test]))

    # prepare grid search models
    scoring, mo = 'r2', 'uniform_average'
    scoring, mo = 'r', 'variance_weighted'
    max_comp = min(X.shape[1], Y.shape[1])
    comp_sweep = np.unique(np.floor(np.linspace(1, max_comp, 20)))

    grid_cca = GridSearchCV(CCA(scoring=scoring, multioutput=mo),
                            dict(n_components=comp_sweep.astype(int)),
                            cv=5)

    grid_regcca = GridSearchCV(RegCCA(n_components=-1,
                                      scoring=scoring, multioutput=mo),
                               dict(alpha=np.logspace(-4, 4, 20)), cv=5)

    models = dict(grid_cca=grid_cca, grid_regcca=grid_regcca)

    for name, model in models.items():
        model.fit(X[train], Y[train])
        print(name, model.best_estimator_.score(X[test], Y[test]))
