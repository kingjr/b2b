from sklearn.preprocessing import scale
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from tqdm import trange
from models import ReducedRankRegressor
import seaborn as sns
import pandas as pd


class JRR(object):
    def __init__(self, G=None, H=None, n_splits=10):
        self.G = RidgeCV() if G is None else G
        self.H = RidgeCV() if H is None else H
        self.n_splits = n_splits

    def fit(self, X, Y):
        H = list()
        for _ in range(self.n_splits):
            p = np.random.permutation(range(len(X)))
            set1, set2 = p[::2], p[1::2]
            self.G.fit(Y[set1], X[set1])
            X_hat = self.G.predict(Y)
            H += [self.H.fit(X[set2], X_hat[set2]).coef_, ]

        # new step to allow predict Y from X
        self.E_ = np.diag(np.diag(np.mean(H, 0)))
        self.G.fit(X @ self.E_, Y)
        return self

    def predict(self, X):
        return self.G.predict(X @ self.E_)


results = list()
for repeat in trange(10):
    n_samples = 1000  # number of samples
    dim_x = 30
    dim_y = 30
    nc = 5  # number of selected components
    snr = .5  # signal to noise ratio

    def make_cov(dim_x):
        Cx = np.random.randn(dim_x, dim_x)
        Cx = Cx.dot(Cx.T) / dim_x  # sym pos-semidefin
        return Cx

    Cx1 = make_cov(dim_x)  # feature covariance in environments 1
    Cn1 = make_cov(dim_x)  # noise covariance in environments 1
    Cx2 = make_cov(dim_x)  # feature covariance in environments 2
    Cn2 = make_cov(dim_x)  # noise covariance in environments 1

    E = np.array([0] * (dim_x - nc) + [1] * (nc))
    np.random.shuffle(E)
    E = np.diag(E)

    # Observation matrix
    F = np.random.randn(dim_x, dim_y)

    def make_data(n_samples, Cx, Cn, E, F, snr):
        X = scale(np.random.multivariate_normal(
            np.zeros(dim_x), Cx, n_samples))
        N = np.random.multivariate_normal(np.zeros(dim_x), Cn, n_samples)
        Y = scale((X @ E * snr + N) @ F)
        return X, Y

    # train set
    X1a, Y1a = make_data(n_samples, Cx1, Cn1, E, F, snr)

    # test set: iid
    X1b, Y1b = make_data(n_samples, Cx1, Cn1, E, F, snr)

    # validation set: new X environments (and/or new noise, both work)
    X2, Y2 = make_data(n_samples, Cx2, Cn2, E, F, snr)

    # Models
    alphas = np.logspace(-3, 3, 10)
    models = dict(
        ridge=RidgeCV(),
        cca=GridSearchCV(CCA(), dict(n_components=[1, 2, 4, 8, 16])),
        pls=GridSearchCV(PLSRegression(), dict(n_components=[1, 2, 4, 8, 16])),
        rrr=GridSearchCV(ReducedRankRegressor(), dict(reg=alphas)),
        jrr=JRR(RidgeCV(alphas), RidgeCV(alphas)),
      )

    # Run
    for name, model in models.items():
        model.fit(X1a, Y1a)
        train, _ = pearsonr(model.predict(X1a).ravel(), Y1a.ravel())
        test, _ = pearsonr(model.predict(X1b).ravel(), Y1b.ravel())
        val, _ = pearsonr(model.predict(X2).ravel(), Y2.ravel())
        results.extend([dict(model=name, score=train, condition='train'),
                        dict(model=name, score=test, condition='test'),
                        dict(model=name, score=val, condition='val')])
results = pd.DataFrame(results)

sns.barplot(x='condition', y='score', data=results, hue='model')
plt.legend(ncol=2)
