import mne
import numpy as np
import os.path as op
from scipy.linalg import svd
from scipy.stats import pearsonr

from sklearn.base import clone
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit


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


def r_score(Y_true, Y_pred):
    """column-wise correlation coefficients"""
    if Y_true.ndim == 1:
        Y_true = Y_true[:, None]
    if Y_pred.ndim == 1:
        Y_pred = Y_pred[:, None]
    R = np.zeros(Y_true.shape[1])
    for idx, (y_true, y_pred) in enumerate(zip(Y_true.T, Y_pred.T)):
        R[idx] = pearsonr(y_true, y_pred)[0]
    return R


class B2B():
    def __init__(self, alphas=np.logspace(-4, 4, 20),
                 independent_alphas=True, ensemble=None):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.ensemble = ensemble

    def fit(self, X, Y):
        
        self.G_ = list()
        self.H_ = list()
        self.Ks_ = list()
        
        # Prepare ensembling
        if self.ensemble is None:
            ensemble = [(range(len(X)), range(len(X))),]
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
            
            # Fit knock-out encoders
            Ks = fit_knockout(X, YG, 
                              self.alphas,
                              self.independent_alphas, True)
            self.Ks_.append(Ks)

        # Aggregate ensembling
        self.G_ = np.mean(self.G_, 0)
        self.H_ = np.mean(self.H_, 0)
        self.Ks_ = np.mean(self.Ks_, 0)
        
        self.E_ = np.diag(self.H_)
        
        return self

    def score(self, X, Y):
        # Transform with decoder
        YG = Y @ self.G_.T
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column X
        return r_score(YG, XH)

    def score_knockout(self, X, Y):
        # Transform with decoder
        YG = Y @ self.G_.T

        # For each knock-out, compute R score
        K_scores = list()
        for xi, K in enumerate(self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            K_scores.append(r_score(YG[:, xi], XK[:, xi])[0])

        return np.array(K_scores)


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

    def score(self, X, Y):
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column of Y
        return r_score(Y, XH)

    def score_knockout(self, X, Y):
        K_scores = list()
        for xi, K in enumerate(self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            K_score = r_score(Y, XK)
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

    def score(self, X, Y):
        # Make standard and knocked-out encoders predictions
        YH = Y @ self.H_.T
        # Compute R for each column of Y
        return r_score(X, YH)

    def score_knockout(self, X, Y):
        return 0
    
    
    def predidct(self, X):
        return 0


def cross_decomp_transform(model, X, Y):
    
        X = X - model.x_mean_
        X /= model.x_std_
        X = X @ model.x_rotations_
        
        Y = Y - model.y_mean_
        Y /= model.y_std_
        Y = Y @ model.y_rotations_
        
        return X, Y


class CrossDecomp():
    """fit X => Y"""
    def __init__(self, model):
        self.model = model

    def fit(self, X, Y):
        # fit CCA
        self.model.fit(X, Y)
        
        if hasattr(self.model, 'estimator'):
            self.model_ = self.model.estimator.fit(X, Y)
        else:
            self.model_ = self.model

        # fit_knockout
        n_x = X.shape[1]
        self.Ks_ = list()
        for xi in np.arange(n_x):
            
            knockout = np.diag(np.ones(n_x))
            knockout[xi] = 0
            self.Ks_.append(clone(self.model_).fit(X @ knockout, Y))
        
        self.E_ = np.sum(self.model_.x_rotations_**2, 1)
        return self

    def score(self, X, Y):
        X, Y = cross_decomp_transform(self.model_, X, Y)
        R_score = r_score(X, Y)
        return R_score
   
    def score_knockout(self, X, Y):

        n_x = X.shape[1]
        
        K_scores = list()
        for xi, K in enumerate(self.Ks_):
       
            knockout = np.diag(np.ones(n_x))
            knockout[xi] = 0
            
            Xk, Yk = cross_decomp_transform(K, X @ knockout, Y)

            K_scores.append(r_score(Xk, Yk))
        
        return np.array(K_scores)
    
    def predict(self, X):
        return self.model_.predict(X)
