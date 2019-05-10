import numpy as np
from scipy.linalg import pinv, svd
from sklearn.metrics import check_scoring
from sklearn.linear_model import RidgeCV
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.ridge import _RidgeGCV, _BaseRidgeCV


class PretrainedRidgeCV(RidgeCV, _RidgeGCV, _BaseRidgeCV):

    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=False,
                 normalize=False, scoring='r2', copy_X=True,
                 store_cv_values=False, multi_alpha=False, X=None):
        """Identical to sklearn RidgeCV, except

        X: can pass X matrix to precompute svd on larger data
        multi_alpha : can identify different alpha for each columns of Y

        Does not yet work in eigen mode
        BUG: scoring cannot be None?
        FIXME: pre computation may be optimized
        TODO: add fit_intercept and sample weights
        """
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        if fit_intercept:
            raise NotImplementedError
        self.normalize = normalize
        self.scoring = scoring
        self.copy_X = copy_X
        self.store_cv_values = store_cv_values
        self.multi_alpha = multi_alpha
        if X is not None:
            U, v, vh = self._decompose_X(X)
            self._v = v
            # FIXME redo pinv ?
            self._inv_vhs = pinv(vh) @ np.diag(1./np.sqrt(v))
            self._prefit = True
        else:
            self._prefit = False

    def _decompose_X(self, X):
        U, s, vh = svd(X, full_matrices=0)
        v = s ** 2
        return U, v, vh

    def _solve(self, alpha, y, v, U, UT_y):
        if alpha > 0:
            alpha_minus_1 = (alpha ** -1)
        elif alpha == 0:
            alpha_minus_1 = 0
        else:
            raise ValueError
        w = ((v + alpha) ** -1) - alpha_minus_1

        c = np.dot(U, w[:, None] * UT_y) + alpha_minus_1 * y
        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        G_diag = (w * U ** 2).sum(axis=-1) + alpha_minus_1
        if len(y.shape) != 1:
            # handle case where y is 2-d
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c

    def fit(self, X, y):
        from sklearn.utils import check_X_y
        from sklearn.utils.extmath import safe_sparse_dot

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'],
                         dtype=[np.float64],
                         multi_output=True, y_numeric=True)

        # decompose
        if self._prefit:
            v = self._v
            U = X @ self._inv_vhs
        else:
            U, v, _ = self._decompose_X(X)
        UT_y = np.dot(U.T, y)

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None

        if np.any(self.alphas <= 0):
            raise ValueError(
                "alphas must be positive. Got {} containing some "
                "negative or null value instead.".format(self.alphas))

        n_samples, n_features = X.shape

        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X)

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        cv_values = np.zeros((n_samples, n_y, len(self.alphas)),
                             dtype=X.dtype)
        C = []
        for i, alpha in enumerate(self.alphas):
            G_diag, c = self._solve(alpha, y, v, U, UT_y)
            if error:
                cv_value = (c / G_diag) ** 2  # squarred error
            else:
                cv_value = y - (c / G_diag)  # prediction
            if cv_value.ndim == 1:
                cv_value = cv_value[..., None]
                c = c[..., None]
            cv_values[..., i] = cv_value
            C.append(c)

        if not self.multi_alpha:
            cv_values = cv_values.reshape(n_samples * n_y, -1)

        if error:
            best = cv_values.mean(axis=0).argmin(axis=-1)
        else:
            # The scorer want an object that will make the predictions but
            # they are already computed efficiently by _RidgeGCV. This
            # identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.decision_function = lambda y_predict: y_predict
            identity_estimator.predict = lambda y_predict: y_predict

            if self.multi_alpha:
                out = np.zeros((len(self.alphas), n_y))
                for i in range(len(self.alphas)):
                    for col in range(n_y):
                        out[i, col] = scorer(identity_estimator, y[:, col],
                                             cv_values[:, col, i])
            else:
                out = [scorer(identity_estimator, y.ravel(), cv_values[:, i])
                       for i in range(len(self.alphas))]
            best = np.argmax(out, axis=0)

        self.alpha_ = self.alphas[best]
        if self.multi_alpha:
            self.dual_coef_ = np.array([C[b][:, col]
                                        for col, b in enumerate(best)]).T
            self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)
        else:
            self.dual_coef_ = C[best]
            self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)

        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = n_samples, len(self.alphas)
            else:
                cv_values_shape = n_samples, n_y, len(self.alphas)
            self.cv_values_ = cv_values.reshape(cv_values_shape)

        return self


if __name__ == '__main__':
    n_samples = 1000
    n_features = 20
    n_chans = 10
    X = np.random.randn(n_samples, n_features)
    W = np.random.randn(n_features, n_chans)
    Y = (X + np.random.randn(n_samples, n_features)) @ W
    alphas = np.logspace(-10, 10)
    ridge = RidgeCV(scoring='r2', alphas=alphas, gcv_mode='svd',
                    fit_intercept=False)
    score_ridge = ridge.fit(X, Y).score(X, Y)

    my_ridge = PretrainedRidgeCV(scoring='r2', alphas=alphas)
    score_myridge = my_ridge.fit(X, Y).score(X, Y)
    assert score_myridge == score_ridge

    for multi_alpha in range(2):
        G = PretrainedRidgeCV(X=Y, multi_alpha=multi_alpha)
        G.fit(Y[1:], X[1:]).score(Y, X)
