import numpy as np
from scipy.linalg import svd
from base import rn_score


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
        U, s, _ = svd(X, full_matrices=0)
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


def fit_knockout(X, Y, alphas, independent_alphas=False,
                 pairwise=False):
    """Fit model using all but knockout X features"""
    n_x, n_y = X.shape[1], Y.shape[1]

    K = list()
    for xi in np.arange(n_x)[:, None]:
        not_xi = np.setdiff1d(np.arange(n_x), xi)

        # When X and Y are matched in dimension, we can speed things
        # up by only computing the regression pairwise, as the other
        # columns won't be analyzed
        if pairwise:
            y_sel = np.asarray(xi)
        else:
            y_sel = np.arange(Y.shape[1])
        x_sel = np.array(not_xi)

        K_ = np.zeros((n_y, n_x))
        K_[y_sel[:, None], x_sel] = ridge_cv(X[:, x_sel], Y[:, y_sel],
                                             alphas, independent_alphas)[0]
        K.append(K_)
    return K


class Forward():
    def __init__(self, alphas=np.logspace(-4, 4, 20), independent_alphas=True):
        self.alphas = alphas
        self.independent_alphas = independent_alphas

    def fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, Y, self.alphas,
                                       self.independent_alphas)
        # Fit knock-out encoders
        self.Ks_ = fit_knockout(X, Y, self.alphas,
                                self.independent_alphas, False)

        self.E_ = np.sum(self.H_**2, 0)
        return self

    def score(self, X, Y, scoring='r', multioutput='raw'):
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column of Y
        return rn_score(Y, XH, scoring=scoring, multioutput=multioutput)

    def score_knockout(self, X, Y, scoring='r', multioutput='raw'):
        K_scores = list()
        for xi, K in enumerate(self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            K_score = rn_score(Y, XK, scoring=scoring, multioutput=multioutput)
            # Difference between standard and knocked-out scores
            K_scores.append(K_score)

        return np.array(K_scores)

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

    def score_knockout(self, X, Y, scoring='r', multioutput='raw'):
        return 0

    def predict(self, X):
        return 0
