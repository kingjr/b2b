import numpy as np
from scipy import linalg
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA as SkCCA
from sklearn.cross_decomposition import PLSRegression as SkPLS
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin


def r_score(X, Y, multioutput='uniform_average'):
    """column-wise correlation coefficients"""

    assert multioutput in ('raw', 'uniform_average', 'variance_weighted')
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    assert len(X) >= 2
    np.testing.assert_equal(X.shape, Y.shape)

    R = np.zeros(X.shape[1])
    for idx, (x, y) in enumerate(zip(X.T, Y.T)):
        R[idx] = pearsonr(x, y)[0]

    if multioutput == 'uniform_average':
        R = R.mean()
    elif multioutput == 'variance_weighted':
        std = np.r_[X, Y].std(0)
        R = np.average(R, weights=std)
    return R


def rn_score(X, Y, scoring='r', multioutput='uniform_average'):
    assert scoring in ('r', 'r2')
    assert multioutput in ('raw', 'uniform_average', 'variance_weighted')

    if scoring == 'r':
        return r_score(X, Y, multioutput=multioutput)

    elif scoring == 'r2':
        return r2_score(X, Y, multioutput=multioutput)


class B2B(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=np.logspace(-4, 4, 20),
                 independent_alphas=True, ensemble=None):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.ensemble = ensemble

    def fit(self, X, Y):

        self.G_ = list()
        self.H_ = list()

        # Prepare ensembling
        if self.ensemble is None:
            ensemble = [(range(len(X)), range(len(X))), ]
        else:
            if isinstance(self.ensemble, int):
                ensemble = ShuffleSplit(self.ensemble)
            else:
                ensemble = self.ensemble
            ensemble = [split for split in ensemble.split(X)]

        # Ensembling loop
        for train, test in ensemble:

            # Fit decoder
            G, G_alpha, YG = ridge_cv(Y[train], X[train],
                                      self.alphas,
                                      self.independent_alphas)
            self.G_.append(G)

            if len(X[train]) != len(X):
                YG = Y @ G.T

            # Fit encoder
            H, H_alpha, _ = ridge_cv(X[test], YG[test],
                                     self.alphas,
                                     self.independent_alphas)
            self.H_.append(H)

        # Aggregate ensembling
        self.G_ = np.mean(self.G_, 0)
        self.H_ = np.mean(self.H_, 0)
        self.E_ = np.diag(self.H_)

        return self

    def fit_H(self, X, Y):

        assert hasattr(self, 'G_')

        YG = Y @ self.G_.T
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, YG,
                                       self.alphas,
                                       self.independent_alphas)

        # Aggregate ensembling
        self.E_ = np.diag(self.H_)
        return self

    def score(self, X, Y, scoring='r', multioutput='raw'):
        if multioutput != 'raw':
            raise NotImplementedError
        # Transform with decoder
        YG = Y @ self.G_.T
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column X
        return rn_score(YG, XH, scoring=scoring, multioutput='raw')


def ridge_cv(X, Y, alphas, independent_alphas=False, Uv=None):
    """ Similar to sklearn RidgeCV but
   (1) can optimize a different alpha for each column of Y
   (2) return leave-one-out Y_hat
   """
    if isinstance(alphas, (float, int)):
        alphas = np.array([alphas, ], np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, n_x = X.shape
    n, n_y = Y.shape
    # Decompose X
    if Uv is None:
        U, s, _ = linalg.svd(X, full_matrices=0)
        v = s**2
    else:
        U, v = Uv
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
    if independent_alphas:
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
    return coefs, best_alphas, Y_hat


class Forward():
    def __init__(self, alphas=np.logspace(-4, 4, 20), independent_alphas=True):
        self.alphas = alphas
        self.independent_alphas = independent_alphas

    def fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, Y, self.alphas,
                                       self.independent_alphas)

        self.E_ = np.sum(self.H_**2, 0)
        return self

    def score(self, X, Y, scoring='r', multioutput='raw'):
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column of Y
        return rn_score(Y, XH, scoring=scoring, multioutput=multioutput)

    def predict(self, X):
        return X @ self.H_.T


class Backward():
    def __init__(self, alphas=np.logspace(-4, 4, 20), independent_alphas=True):
        self.alphas = alphas
        self.independent_alphas = independent_alphas

    def fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(Y, X, self.alphas,
                                       self.independent_alphas)

        self.E_ = np.sum(self.H_**2, 1)
        return self

    def score(self, X, Y, scoring='r', multioutput='raw'):
        # Make standard and knocked-out encoders predictions
        YH = Y @ self.H_.T
        # Compute R for each column of Y
        return rn_score(X, YH, scoring=scoring, multioutput=multioutput)

    def predict(self, X):
        return 0


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
        self.E_ = np.sum(self.x_rotations_**2, 1)
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
        self.E_ = np.sum(self.x_rotations_**2, 1)
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
        self.E_ = np.sum(self.x_rotations_**2, 1)
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
