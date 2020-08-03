import numpy as np
from scipy import linalg
from copy import deepcopy
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA as SkCCA
from sklearn.cross_decomposition import PLSRegression as SkPLS
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.simplefilter("ignore")

alphas = np.logspace(-4, 4, 20)
components = np.linspace(0., 1., 20)


def r_score(X, Y, multioutput='uniform_average'):
    """column-wise correlation coefficients"""

    assert multioutput in ('raw', 'raw_values', 'uniform_average',
                           'variance_weighted')
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
    assert multioutput in ('raw', 'raw_values',
                           'uniform_average', 'variance_weighted')

    if scoring == 'r':
        return r_score(X, Y, multioutput=multioutput)

    elif scoring == 'r2':
        return r2_score(X, Y, multioutput=multioutput)


class B2B(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=alphas,
                 independent_alphas=True, ensemble=None,
                 G='ridge_cv',
                 H='ridge_cv',
                 scoring='r'):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.ensemble = ensemble
        self.scoring = scoring
        self.__name__ = 'B2B'
        self.G = G
        self.H = H

    def fit(self, X, Y):
        import copy

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
            if self.G == 'ridge_cv':
                G, G_alpha, YG = ridge_cv(Y[train], X[train],
                                          self.alphas,
                                          self.independent_alphas)
                if len(X[train]) != len(X):
                    YG = Y @ G.T
            else:
                G = copy.deepcopy(self.G).fit(Y[train], X[train])
                YG = np.empty((len(Y), X.shape[1]))
                YG[test] = G.predict(Y[test])

            self.G_.append(G)

            # Fit encoder
            if self.H == 'ridge_cv':
                H, H_alpha, _ = ridge_cv(X[test], YG[test],
                                         self.alphas,
                                         self.independent_alphas)
            else:
                H = copy.deepcopy(self.H).fit(X[test], YG[test])
            self.H_.append(H)

        # Aggregate ensembling
        if self.G == 'ridge_cv':
            self.G_ = np.mean(self.G_, 0)
        if self.H == 'ridge_cv':
            self.H_ = np.mean(self.H_, 0)
            self.S_ = np.diag(self.H_)
        else:
            if hasattr(self.H_[0], 'coef_'):
                self.H_ = np.mean([h.coef_ for h in self.H_], 0)
                self.S_ = np.diag(self.H_)
            else:
                self.S_ = None

        return self

    def fit_H(self, X, Y):

        assert hasattr(self, 'G_')

        YG = Y @ self.G_.T
        # Fit encoder
        if self.H == 'ridge_cv':
            self.H_, H_alpha, _ = ridge_cv(X, YG,
                                           self.alphas,
                                           self.independent_alphas)
        else:
            raise NotImplementedError

        # Aggregate ensembling
        self.S_ = np.diag(self.H_)
        return self

    def score(self, X, Y, scoring=None, multioutput='raw_values'):
        scoring = self.scoring if scoring is None else scoring
        if multioutput != 'raw_values':
            raise NotImplementedError
        if self.G == 'ridge_cv':
            # Transform with decoder
            YG = Y @ self.G_.T
        else:
            YG = np.mean([G.predict(Y) for G in self.G_], 0)

        # Make standard and knocked-out encoders predictions
        if self.H == 'ridge_cv':
            XH = X @ self.H_.T
        else:
            XH = np.mean([H.predict(X) for H in self.H_], 0)
        # Compute R for each column X
        return rn_score(YG, XH,
                        scoring=scoring, multioutput='raw_values')


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
    def __init__(self, alphas=alphas, independent_alphas=True,
                 scoring='r', multioutput='uniform_average'):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.scoring = scoring
        self.multioutput = multioutput
        self.__name__ = 'Forward'

    def fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, Y, self.alphas,
                                       self.independent_alphas)

        self.S_ = np.sum(self.H_**2, 0)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column of Y
        return rn_score(Y, XH, scoring=scoring, multioutput=multioutput)

    def predict(self, X):
        return X @ self.H_.T


class Backward():
    def __init__(self, alphas=alphas, independent_alphas=True,
                 scoring='r'):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.scoring = scoring
        self.__name__ = 'Backward'

    def fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(Y, X, self.alphas,
                                       self.independent_alphas)

        self.S_ = np.sum(self.H_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput='raw_values'):
        scoring = self.scoring if scoring is None else scoring
        if multioutput != 'raw_values':
            raise NotImplementedError
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


def validate_number_components(n, X, Y):
    n_max = min(X.shape[1], Y.shape[1])
    if n == -1:
        n = n_max
    elif n >= 0. and n < 1.:
        n = int(np.floor(n_max * n))
        n = 1 if n == 0 else n

    assert n == int(n) and n > 0 and n <= n_max
    return int(n)


class GridPLS(BaseEstimator, RegressorMixin):
    """Optimize n_components by minimizing Y_pred error"""

    def __init__(self, n_components=components, cv=5,
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.n_components = n_components
        self.cv = cv
        self.scoring = scoring
        self.multioutput = multioutput
        self.__name__ = 'GridPLS'

    def fit(self, X, Y):

        N = self.n_components
        if not isinstance(N, (list, np.ndarray)):
            N = [N, ]

        components = np.unique([validate_number_components(n, X, Y)
                                for n in N])
        # Optimize n_components on Y prediction!
        if len(components) > 1:
            models = GridSearchCV(SkPLS(), dict(n_components=components))
            best = models.fit(X, Y).best_estimator_
            self.n_components_ = best.n_components

            x_valid = range(X.shape[1])
            y_valid = range(Y.shape[1])
        else:
            best = PLS(n_components=components[0],
                       scoring=self.scoring,
                       multioutput=self.multioutput)
            best.fit(X, Y)
            self.n_components_ = best.n_components_
            x_valid = best.x_valid_
            y_valid = best.y_valid_

        self.x_mean_ = np.zeros(X.shape[1])
        self.x_std_ = np.zeros(X.shape[1])
        self.x_rotations_ = np.zeros((X.shape[1], self.n_components_))
        self.y_mean_ = np.zeros(Y.shape[1])
        self.y_std_ = np.zeros(Y.shape[1])
        self.y_rotations_ = np.zeros((Y.shape[1], self.n_components_))

        self.x_mean_[x_valid] = best.x_mean_
        self.x_std_[x_valid] = best.x_std_
        self.x_rotations_[x_valid, :] = best.x_rotations_
        self.y_mean_[y_valid] = best.y_mean_
        self.y_std_[y_valid] = best.y_std_
        self.y_rotations_[y_valid, :] = best.y_rotations_

        self.S_ = np.sum(self.x_rotations_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)

    def transform(self, X):
        return self.best.transform(X[:, self.x_valid_])

    def fit_transform(self, X, Y):
        return self.fit(X, Y).best.transform(X)


class GridCCA(BaseEstimator, RegressorMixin):
    """Optimize n_components by minimizing Y_pred error"""

    def __init__(self, n_components=components, cv=5,
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.n_components = n_components
        self.cv = cv
        self.scoring = scoring
        self.multioutput = multioutput
        self.__name__ = 'GridCCA'

    def fit(self, X, Y):

        N = self.n_components
        if not isinstance(N, (list, np.ndarray)):
            N = [N, ]
        components = np.unique([validate_number_components(n, X, Y)
                                for n in N])
        # Optimize n_components on Y prediction!
        if len(components) > 1:
            models = GridSearchCV(SkCCA(), dict(n_components=components))
            best = models.fit(X, Y).best_estimator_
            self.n_components_ = best.n_components
            x_valid = range(X.shape[1])
            y_valid = range(Y.shape[1])
        else:
            best = CCA(n_components=components[0],
                       scoring=self.scoring,
                       multioutput=self.multioutput)
            best.fit(X, Y)
            self.n_components_ = best.n_components_

            x_valid = best.x_valid_
            y_valid = best.y_valid_

        self.x_mean_ = np.zeros(X.shape[1])
        self.x_std_ = np.zeros(X.shape[1])
        self.x_rotations_ = np.zeros((X.shape[1], self.n_components_))
        self.y_mean_ = np.zeros(Y.shape[1])
        self.y_std_ = np.zeros(Y.shape[1])
        self.y_rotations_ = np.zeros((Y.shape[1], self.n_components_))

        self.x_mean_[x_valid] = best.x_mean_
        self.x_std_[x_valid] = best.x_std_
        self.x_rotations_[x_valid, :] = best.x_rotations_
        self.y_mean_[y_valid] = best.y_mean_
        self.y_std_[y_valid] = best.y_std_
        self.y_rotations_[y_valid, :] = best.y_rotations_

        self.S_ = np.sum(self.x_rotations_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)

    def transform(self, X):
        return self.best.transform(X[:, self.x_valid_])

    def fit_transform(self, X, Y):
        return self.fit(X, Y).best.transform(X)


class GridRegCCA(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=np.logspace(-4, 4., 20), cv=5,
                 n_components=[-1, ],
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.n_components = n_components
        self.multioutput = multioutput
        self.__name__ = 'GridRegCCA'

    def fit(self, X, Y):

        self.x_valid_ = np.where(X.std(0) > 0)[0]
        self.y_valid_ = np.where(Y.std(0) > 0)[0]
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]

        N = self.n_components
        if not isinstance(N, (list, np.ndarray)):
            N = [N, ]
        components = np.unique([validate_number_components(n, X, Y)
                                for n in N])
        grid = {'alpha': self.alphas,
                'n_components': components}

        # Optimize n_components on Y prediction!
        if np.prod(list(map(np.shape, grid.values()))) > 1:
            models = GridSearchCV(RegCCA(scoring=self.scoring,
                                         multioutput=self.multioutput),
                                  grid)
            best = models.fit(X, Y).best_estimator_
        else:
            best = RegCCA(alpha=grid['alpha'][0], n_components=components[0],
                          scoring=self.scoring, multioutput=self.multioutput)
            best.fit(X, Y)
        self.n_components_ = best.n_components
        self.alpha_ = best.alpha

        self.x_mean_ = best.x_mean_
        self.x_std_ = best.x_std_
        self.x_rotations_ = best.x_rotations_
        self.y_mean_ = best.y_mean_
        self.y_std_ = best.y_std_
        self.y_rotations_ = best.y_rotations_

        self.S_ = np.sum(self.x_rotations_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):

        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)

    def transform(self, X):
        return self.best.transform(X[:, self.x_valid_])

    def fit_transform(self, X, Y):
        return self.fit(X, Y).best.transform(X)


class CCA(SkCCA):
    """overwrite scikit-learn CCA to get scoring function in
       canonical space"""

    def __init__(self, n_components=-1,
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.scoring = scoring
        self.multioutput = multioutput
        self.__name__ = 'CCA'
        super().__init__(n_components=n_components, tol=tol)

    def fit(self, X, Y):

        self.x_valid_ = np.where(X.std(0) > 0)[0]
        self.y_valid_ = np.where(Y.std(0) > 0)[0]
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]

        N = self.n_components
        self.n_components = validate_number_components(N, X, Y)
        super().fit(X, Y)
        self.n_components_ = self.n_components
        self.n_components = N
        self.S_ = np.sum(self.x_rotations_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)

    def transform(self, X):
        return super().transform(X[:, self.x_valid_])

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)


class PLS(SkPLS):
    """overwrite scikit-learn PLSRegression to get scoring function in
       canonical space"""

    def __init__(self, n_components=-1,
                 scoring='r', multioutput='uniform_average', tol=1e-15):
        self.scoring = scoring
        self.multioutput = multioutput
        self.__name__ = 'PLS'
        super().__init__(n_components=n_components, tol=tol)

    def fit(self, X, Y):
        self.x_valid_ = np.where(X.std(0) > 0)[0]
        self.y_valid_ = np.where(Y.std(0) > 0)[0]
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]
        N = self.n_components
        self.n_components = validate_number_components(N, X, Y)
        super().fit(X, Y)
        self.n_components_ = self.n_components
        self.n_components = N
        self.S_ = np.sum(self.x_rotations_**2, 1)
        return self

    def score(self, X, Y, scoring=None, multioutput=None):
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]
        scoring = self.scoring if scoring is None else scoring
        multioutput = self.multioutput if multioutput is None else multioutput
        return canonical_correlation(self, X, Y,
                                     scoring, multioutput)

    def transform(self, X):
        return super().transform(X[:, self.x_valid_])

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)


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
        self.__name__ = 'RegCCA'

    def fit(self, X, Y):

        self.x_valid_ = np.where(X.std(0) > 0)[0]
        self.y_valid_ = np.where(Y.std(0) > 0)[0]
        X = X[:, self.x_valid_]
        Y = Y[:, self.y_valid_]

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

        X = (X - self.x_mean_) / self.x_std_
        Y = (Y - self.y_mean_) / self.y_std_

        # compute cca
        comps = self._compute_kcca([X, Y],
                                   reg=self.alpha, numCC=dz)
        self.x_rotations_ = comps[0]
        self.y_rotations_ = comps[1]
        self.S_ = np.sum(self.x_rotations_**2, 1)

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
        try:
            r, Vs = linalg.eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
        except linalg.LinAlgError:  # noqa
            r = np.zeros(numCC)
            Vs = np.zeros((sum(nFs), numCC))

        r[np.isnan(r)] = 0
        rindex = np.argsort(r)[::-1]
        comp = []
        Vs = Vs[:, rindex]
        for i in range(nDs):
            comp.append(Vs[sum(nFs[:i]):sum(nFs[:i + 1]), :numCC])
        return comp


def score_knockout(model, X, Y, XY_train=None, scoring='r', fix_grid=True):
    assert isinstance(model, (CCA, PLS, GridCCA, GridPLS, RegCCA,
                              GridRegCCA, B2B, Forward, Backward))
    assert len(X) == len(Y)
    assert scoring in ('r', 'r2')
    is_b2b = isinstance(model, B2B)
    dim_x = X.shape[1]

    # Compute standard scores
    score_full = model.score(X, Y,
                             scoring=scoring,
                             multioutput='raw_values')
    score_delta = np.zeros(dim_x)

    # Compute knock out scores
    for f in range(dim_x):

        # Setup knockout matrix
        knockout = np.eye(dim_x)
        knockout[f] = 0

        model_ = model
        # refit the model
        if XY_train is not None:
            X_train, Y_train = XY_train
            model_ = deepcopy(model)
            if isinstance(model, (GridPLS, GridCCA, GridRegCCA)) and fix_grid:
                n = model.n_components_
                model_.n_components = -1 if n == X.shape[1] else n
            if is_b2b:
                model_.fit_H(X_train @ knockout, Y_train)
            else:
                model_.fit(X_train @ knockout, Y_train)

        # Score
        score_ko = model_.score(X @ knockout, Y,
                                scoring=scoring,
                                multioutput='raw_values')

        # Aggregate predicted dimensions
        if is_b2b:
            score_delta[f] = score_full[f] - score_ko[f]
        elif len(score_full) != len(score_ko):
            print('Different dims!')
            score_delta[f] = score_full.mean() - score_ko.mean()
        else:
            score_delta[f] = (score_full - score_ko).mean()

    return score_delta
