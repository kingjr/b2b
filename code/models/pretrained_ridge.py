import numpy as np
from sklearn.model_selection import KFold
from scipy.linalg import pinv, sqrtm
from sklearn.base import BaseEstimator


class PretrainedRidgeCV(BaseEstimator):
    def __init__(self, Cx, alphas=(.1, 1., 10.), cv=3, non_spherical=False):
        self.Cx = Cx
        if isinstance(alphas, (float, int)):
            alphas = [alphas, ]
        self.alphas = alphas

        # initialize all precisions
        self.Cx_invs = [pinv(Cx + alpha * np.identity(len(Cx)))
                        for alpha in self.alphas]
        self.cv = KFold(cv)
        self.non_spherical = non_spherical
        if non_spherical:
            self.Cx_sqrt = sqrtm(Cx)

    def fit(self, X, Y):
        # Grid search alpha
        loss = np.zeros((self.cv.n_splits, len(self.alphas)))
        for split, (train, test) in enumerate(self.cv.split(X, Y)):
            Cxy = X[train].T @ Y[train]
            for idx, Cx_inv in enumerate(self.Cx_invs):
                coef = Cx_inv @ Cxy
                if self.non_spherical:
                    coef = self.Cx_sqrt @ coef
                Y_hat = X[test] @ coef
                loss[split, idx] = np.sum((Y[test] - Y_hat)**2)

        best_alpha_idx = np.argmin(loss.mean(0))
        self.best_alpha_ = self.alphas[best_alpha_idx]

        # refit
        self.coef_ = self.Cx_invs[best_alpha_idx] @ X.T @ Y
        if self.non_spherical:
            self.coef_ = self.Cx_sqrt @ self.coef_
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
    n_samples = 1000
    n_features = 20
    n_chans = 10
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_chans)
    G = PretrainedRidgeCV(Y.T @ Y)
    H = PretrainedRidgeCV(X.T @ X)
    G.fit(Y, X).score(Y, X)
    H.fit(X, Y).score(X, Y)
