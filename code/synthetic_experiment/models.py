from utils import sonquist_morgan
import numpy as np

def ridge_cv(X,
             Y,
             alphas=np.logspace(-5, 5, 20),
             independent_alphas=True):
    """
    Similar to sklearn RidgeCV but
    (1) can optimize a different alpha for each column of Y (independent_alphas=True)
    (2) return leave-one-out Y_hat
    """
    if isinstance(alphas, (float, int)):
        alphas = np.array([alphas, ], np.float64)
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


def jrr(X, Y):
    _, _, X_hat = ridge_cv(Y, X)
    H, _, _ = ridge_cv(X, X_hat)
    return np.diag(H)

def ols(X, Y):
    H, _, _ = ridge_cv(X, Y)
    return np.diag(H)


class OLS(object):
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.coef = np.linalg.pinv(X.T @ X) @ X.T @ Y

    def predict(self, X):
        return X @ self.coef
    
    def solution(self):
        return sonquist_morgan(np.power(self.coef, 2).sum(1))



