import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from scipy.linalg import pinv, svd
from itertools import product


def ridge_cv(X, Y, alphas, independent=False):
    """
    Similar to sklearn RidgeCV but
    (1) can optimize a different alpha for each column of Y
    (2) return leave-one-out Y_hat
    """
    if isinstance(alphas, (float, int)):
        alphas = np.array([alphas, ], np.float64)
    alphas = np.asarray(alphas)
    n, n_x = X.shape
    n, n_y = Y.shape
    # Decompose X
    U, s, _ = svd(X, full_matrices=0)
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
    if independent:
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
    return coefs, alphas[best_alphas], Y_hat


class RidgeCV():
    def __init__(self, alphas, independent):
        self.alphas = alphas
        self.independent = independent

    def __call__(self, X, Y):
        return ridge_cv(X, Y, self.alphas, self.independent)


def trunkated_ols_cv(X, Y, ranks=None, independent=False):
    """
    solves (X'X)^-1 X'Y with rank optimal constrain on X
    (1) can optimize a different alpha for each column of Y (independent=True)
    (2) return leave-one-out Y_hat
    """
    independent = False
    n, n_x = X.shape
    n, n_y = Y.shape
    # Decompose X
    U, s, _ = svd(X, full_matrices=0)  # U = shape(n, n_x)
    v = s**2

    cv_duals = np.zeros((n_x, n, n_y))
    cv_errors = np.inf * np.ones((n_x, n, n_y))

    ranks = np.arange(n_x) if ranks is None else np.asarray(ranks)

    for rank in ranks:
        # Solve
        # ridge cv: w = ((v + alpha) ** -1) - alpha ** -1
        D = np.eye(n) - U[:, :rank] @ U[:, :rank].T  # n_samples x n_samples
        c = D @ Y  # n_samples x dim_y
        # ridge cv: c = U @ np.diag(w) @ UY + alpha ** -1 * Y

        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        # G_diag = (w * U ** 2).sum(axis=-1) + alpha ** -1
        G_diag = np.diag(D)
        error = c / G_diag[:, np.newaxis]
        cv_errors[rank] = error
        cv_duals[rank] = U[:, :rank] @ np.diag(v[:rank]**-1) @ U[:, :rank].T @ Y # noqa

    # identify best alpha for each column of Y independently
    if independent:
        best_ranks = (cv_errors ** 2).mean(axis=1).argmin(axis=0)
        duals = np.transpose([cv_duals[b, :, i]
                              for i, b in enumerate(best_ranks)])
        cv_errors = np.transpose([cv_errors[b, :, i]
                                  for i, b in enumerate(best_ranks)])
    else:
        _cv_errors = cv_errors.reshape(n_x, -1)
        best_ranks = (_cv_errors ** 2).mean(axis=1).argmin(axis=0)
        duals = cv_duals[best_ranks]
        cv_errors = cv_errors[best_ranks]

    coefs = duals.T @ X
    Y_hat = Y - cv_errors
    return coefs, best_ranks, Y_hat


class TrunkOLSCV():
    def __init__(self, ranks, independent):
        self.ranks = ranks
        self.independent = independent

    def __call__(self, X, Y):
        return trunkated_ols_cv(X, Y, self.ranks, self.independent)


def trunkated_ols(X, Y):
    pca = PCA('mle').fit(X)
    Xt = pca.inverse_transform(pca.transform(X))
    coef = pinv(Xt.T @ Xt) @ Xt.T @ Y
    return coef.T, pca.n_components_, np.zeros_like(Y)


class JRR():
    def __init__(self, G, H, bagging):
        self.G = G
        self.H = H
        assert isinstance(bagging, int)
        self.bagging = bagging
    def __call__(self, X, Y):
        if self.bagging == 0:
            G, best_reg, Y_hat = self.G(Y, X)
            X_hat = Y @ G.T
            H, best_reg, Y_hat = self.H(X, X_hat)
        else:
            Hs = list()
            for _ in range(self.bagging):
                p = np.random.permutation(range(len(X)))
                train, test = p[::2], p[1::2]
                G, _, _ = self.G(Y[train], X[train])
                X_hat = Y[test] @ G.T
                H, _, _ = self.H(X[test], X_hat)
                Hs.append(H)
            H = np.mean(Hs, 0)
        self.E = np.diag(H)
        return self.E


# synthetic data
n = 100  # number of samples
dim_x = 10  # dimensionality of X
dim_y = 11

Cx = np.random.randn(dim_x, dim_x)
Cx = Cx.dot(Cx.T) / dim_x  # sym pos-semidefin

X = np.random.multivariate_normal(np.zeros(dim_x), Cx, n)
N = np.random.randn(n, dim_x)
F = np.random.randn(dim_y, dim_x)
E = np.diag(np.random.rand(dim_x) < 0.05)

Y = (X @ E + N) @ F.T
X, Y, E = scale(X), scale(Y), np.diag(E)

# prepare all possible JRRs

alphas = np.logspace(-5, 5, 30)
g_ranks = np.linspace(1, dim_y-1, 30).astype(int)
h_ranks = np.linspace(1, dim_x-1, 30).astype(int)

G_solvers = (
    ('ols', RidgeCV(1e-6, False)),
    ('ridgecv', RidgeCV(alphas, independent=True)),
    ('rankcv', TrunkOLSCV(g_ranks, independent=True)),
    ('rankmle', trunkated_ols)
)

H_solvers = (
    ('ols', RidgeCV(1e-6, False)),
    ('ridgecv', RidgeCV(alphas, independent=False)),
    ('rankcv', TrunkOLSCV(h_ranks, independent=False)),
    ('rankmle', trunkated_ols)
)


params = (
    G_solvers,  # G
    H_solvers,  # H
    (('loo', 0), ('bag', 10)),  # bagging or leave one out
)

functions = dict()
for (label_g, G), (label_h, H), (label_bag, bagging) in product(*params):
    label = '_'.join((label_g, label_h, label_bag))
    if (label_bag == 'loo') and ('cv' not in label_g or 'cv'):
        continue
    functions[label] = JRR(G=G, H=H, bagging=bagging)


results = dict()
for label, func in functions.items():
    print(label)
    E_hat = func(X, Y)
    results[label] = E_hat
