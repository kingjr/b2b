from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import ShuffleSplit
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt


def make_data(seed=0):
    np.random.seed(seed)

    # make data
    n = 1000  # number of samples
    nX = 60  # dimensionality of X
    nY = 60

    Cx = np.random.randn(nX, nX)
    Cx = Cx.dot(Cx.T) / nX  # sym pos-semidefin

    X = np.random.multivariate_normal(np.zeros(nX), Cx, n)
    N = np.random.randn(n, nX)
    F = np.random.randn(nY, nX)
    E = np.eye(nX)
    E[:10] = 0
    E[30:] = 0

    Y = (X @ E + N) @ F.T
    Y += np.random.randn(*Y.shape)
    return X, Y, E


if __name__ == '__main__':
    X, Y, E = make_data()
    n, nX = X.shape

    alphas = np.logspace(-6, 6, 20)
    ridge = RidgeCV(alphas=alphas, fit_intercept=False)
    ols = LinearRegression(fit_intercept=False)

    # cv
    set1, set2 = range(n//2), range(n//2, n)
    cv = ShuffleSplit(100, test_size=.5, random_state=0)

    X, Y, E = make_data(seed=2)
    
    # vanilla JRR Bias
    E_hats = list()
    for set1, set2 in cv.split(X, Y):
        # decode
        G = ridge.fit(Y[set1], X[set1])
        X_hat = G.predict(Y)
        # Encode
        E_hat = ridge.fit(X[set2], X_hat[set2]).coef_
        E_hats.append(E_hat)
    E_hats = np.mean(E_hats, 0)

    # plot
    vmax = .3
    fig, axes = plt.subplots(1, 2)

    axes[0].fill_between(range(len(E)), np.diag(E_hat))
    not_causal = np.where(np.diag(E) == 0)[0]
    _, p_value = ttest_1samp(np.diag(E_hat)[not_causal], 0)
    axes[0].set_title('non causal > 0: p=%.4f' % p_value)
    if vmax is None:
        vmax = np.abs(np.percentile(np.diag(E_hat)[not_causal], 90)) * 2.
    axes[0].set_ylim(-vmax, vmax)
    axes[1].matshow(E_hat, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
