import numpy as np
from scipy.linalg import svd
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone


def ridge_cv(X, Y, alphas, independent_alphas=False, Uv=None):
    """ Similar to sklearn RidgeCV but
   (1) can optimize a different alpha for each column of Y
   (2) return leave-one-out Y_hat
   """
    if isinstance(alphas, (float, int)):
        alphas = np.array([
            alphas,
        ], np.float64)
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
        w = ((v + alpha)**-1) - alpha**-1
        c = U @ np.diag(w) @ UY + alpha**-1 * Y
        cv_duals[alpha_idx] = c

        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        G_diag = (w * U**2).sum(axis=-1) + alpha**-1
        error = c / G_diag[:, np.newaxis]
        cv_errors[alpha_idx] = error

    # identify best alpha for each column of Y independently
    if independent_alphas:
        best_alphas = (cv_errors**2).mean(axis=1).argmin(axis=0)
        duals = np.transpose(
            [cv_duals[b, :, i] for i, b in enumerate(best_alphas)])
        cv_errors = np.transpose(
            [cv_errors[b, :, i] for i, b in enumerate(best_alphas)])
    else:
        _cv_errors = cv_errors.reshape(len(alphas), -1)
        best_alphas = (_cv_errors**2).mean(axis=1).argmin(axis=0)
        duals = cv_duals[best_alphas]
        cv_errors = cv_errors[best_alphas]

    coefs = duals.T @ X
    Y_hat = Y - cv_errors
    return coefs, best_alphas, Y_hat


def fit_knockout(X,
                 Y,
                 alphas,
                 knockout=True,
                 independent_alphas=False,
                 pairwise=False):
    """Fit model using all but knockout X features"""
    n_x, n_y = X.shape[1], Y.shape[1]
    knockout = np.arange(n_x)[:, None] if knockout is True else knockout

    K = list()
    for xi in knockout:
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
        K_[y_sel[:, None], x_sel] = ridge_cv(X[:, x_sel], Y[:, y_sel], alphas,
                                             independent_alphas)[0]
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
    def __init__(self,
                 alphas=np.logspace(-4, 4, 20),
                 independent_alphas=True,
                 knockout=True,
                 ensemble=None):
        self.alphas = alphas
        self.knockout = knockout
        self.independent_alphas = independent_alphas
        self.ensemble = ensemble

    def _fit(self, X, Y):
        self.G_ = list()
        self.H_ = list()
        self.Ks_ = list()
        if self.ensemble is None:
            ensemble = [
                (range(len(X)), range(len(X))),
            ]
        else:
            if isinstance(self.ensemble, int):
                ensemble = ShuffleSplit(self.ensemble)
            else:
                ensemble = self.ensemble
            ensemble = [split for split in ensemble.split(X)]
        for train, test in ensemble:
            # Fit decoder
            G, _, _ = ridge_cv(Y[train], X[train], self.alphas,
                               self.independent_alphas)
            self.G_.append(G)

            YG = Y[test] @ G.T
            H, _, _ = ridge_cv(X[test], YG, self.alphas,
                               self.independent_alphas)
            self.H_.append(H)

            # Fit knock-out encoders
            if self.knockout:
                Ks = fit_knockout(X[test], YG, self.alphas, self.knockout,
                                  self.independent_alphas, True)
                self.Ks_.append(Ks)
        self.G_ = np.mean(self.G_, 0)
        self.H_ = np.mean(self.H_, 0)
        if self.knockout:
            self.Ks_ = np.mean(self.Ks_, 0)
        return self

    def fit(self, X, Y):
        # for held out prediction evaluation
        train = range(len(X) // 2)
        test = range(len(X) // 2, len(X))
        self._fit(X[train], Y[train])
        self.R_score_ = self._score(X[test], Y[test])

        # for predictions
        self._fit(X, Y)
        self.coef, _, _ = ridge_cv(X @ np.diag(np.diag(self.H_)), Y,
                                   self.alphas)

    def predict(self, X):
        return X @ np.diag(np.diag(self.H_)) @ self.coef

    def _score(self, X, Y):
        """return the specific contribution of feature i:
        R(XH, YG) - R(Xk Hk, YG)

        where k indicate all hypothetical causes but i
        """
        # Transform Y with decoder
        YG = Y @ self.G_.T

        # compute score with complete model
        R_score = r_score(YG, X @ self.H_.T)

        # compute score with all but one feature (a.k.a knockout score)
        # Allow knocking out blocks of features
        if self.knockout is True:
            knockout = range(X.shape[1])
        else:
            knockout = self.knockout

        # For each knock-out, compute R score
        K_scores = list()
        for xi, K in zip(knockout, self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            r = r_score(YG[:, xi], XK[:, xi])
            # If one dimension, make it float
            if isinstance(xi, int):
                r = r[0]
            # Difference between standard and knocked-out scores
            K_scores.append(r)

        return R_score - K_scores

    def solution(self):
        return self.R_score_


class Forward():
    def __init__(self,
                 alphas=np.logspace(-4, 4, 20),
                 independent_alphas=False,
                 knockout=True):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.knockout = knockout

    def _fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, Y, self.alphas,
                                       self.independent_alphas)
        # Fit knock-out encoders
        if self.knockout:
            self.Ks_ = fit_knockout(X, Y, self.alphas, self.knockout,
                                    self.independent_alphas, False)

        return self

    def fit(self, X, Y):
        # for held out prediction evaluation
        train = range(len(X) // 2)
        test = range(len(X) // 2, len(X))
        self._fit(X[train], Y[train])
        self.R_score_ = self._score(X[test], Y[test])

        # for predictions
        self._fit(X, Y)

    def predict(self, X):
        return X @ self.H_.T

    def _score(self, X, Y):
        # Compute R for each column of Y
        R_scores = r_score(self.predict(X), Y)

        # Compute R-score for each knockout model (a.k.a  K-score)
        if self.knockout is True:
            knockout = range(X.shape[1])
        else:
            knockout = self.knockout

        K_scores = list()
        for xi, K in zip(knockout, self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            K_score = r_score(Y, XK)
            # Difference between standard and knocked-out scores
            K_scores.append(K_score)

        return R_scores - K_scores

    def solution(self, ):
        return np.sum(self.H_**2, 0)


class CrossDecomp():
    def __init__(self,
                 G=CCA(2),
                 alphas=np.logspace(-4, 4, 20),
                 independent_alphas=False,
                 knockout=True):
        self.G = G
        self.H = Forward(alphas, independent_alphas, knockout)

    def _fit(self, X, Y):
        self.G.fit(Y, X)
        YG = self.G.transform(Y)
        self.H.fit(X, YG)
        return self

    def fit(self, X, Y):
        # for held out prediction evaluation
        train = range(len(X) // 2)
        test = range(len(X) // 2, len(X))
        self._fit(X[train], Y[train])
        self.R_score_ = self._score(X[test], Y[test])

        # for predictions
        self._fit(X, Y)
        self.invG = clone(self.G).fit(X, Y)

    def predict(self, X):
        return self.invG.predict(X)

    def _score(self, X, Y):
        YG = self.G.transform(Y)
        return self.H._score(X, YG)

    def solution(self, ):
        return np.sum(self.R_score_**2, 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def make_data(snr=1):
        dx = 20
        dy = 20
        n = 1000
        e = dx // 2
        E = np.diag(np.ones(dx))
        E[e:] = 0
        Cx = np.random.randn(dx, dx)
        X = np.random.multivariate_normal(np.zeros(dx), Cx, n)
        # make source noise move in different dimension than causal factors
        Nx = np.random.randn(n, dx)
        Nx[:, :e] = 0
        Ny = np.random.randn(n, dy)
        F = np.random.randn(dx, dy)
        Y = (X @ E + Nx / snr) @ F + Ny
        return scale(X), scale(Y), np.diag(E)

    X, Y, e = make_data(snr=.25)

    # let's give CCA a chance and already give the right amount of dimensions
    n_comp = int(np.sum(e))
    models = dict(
        Forward=Forward(),
        B2B=B2B(),
        CCA=CrossDecomp(CCA(n_comp)),
        PLS=CrossDecomp(PLSRegression(n_comp)))

    # model parameters
    lines = list()
    for name, model in models.items():
        model.fit(X, Y)
        sol = model.solution()
        plt.plot(sol, label=name)
    plt.legend()
    plt.title('model solution (y axis not comparable)')
    plt.ylabel('R or sum(R^2)')
    plt.xlabel('X')
