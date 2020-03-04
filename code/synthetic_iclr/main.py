import numpy as np
from models import B2B, Forward, Backward, GridCCA, GridPLS, GridRegCCA
from models import score_knockout
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
# from tqdm import trange
import time
# import pandas as pd
# from tqdm import trange
# from itertools import product
# import pickle
# import submitit
# import seaborn as sns
# import matplotlib.pyplot as plt


class Synthetic(object):
    def __init__(self,
                 dim_x=50,         # number of features
                 dim_y=30,         # number of sensors
                 nc=5,             # number of selected features
                 snr=1.0,          # signal-to-noise ratio
                 nonlinear=False):  # number of selected features

        # linear transformation
        self.F = np.random.randn(dim_x, dim_y) / np.sqrt(dim_x)

        # masking transformation
        self.E = np.array([0] * (dim_x - nc) + [1] * (nc))

        # features covariance
        self.cov_X = np.random.randn(dim_x, dim_x) / np.sqrt(dim_x)
        self.cov_X = self.cov_X @ self.cov_X.T

        # noise covariance
        self.cov_N = np.random.randn(dim_x, dim_x) / np.sqrt(dim_x)
        self.cov_N = self.cov_N @ self.cov_N.T

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nonlinear = nonlinear
        self.snr = snr

    def sample(self, n_samples=1000):
        X = np.random.multivariate_normal(np.zeros(self.dim_x),
                                          self.cov_X, n_samples)
        N = np.random.multivariate_normal(np.zeros(self.dim_x),
                                          self.cov_N, n_samples)

        # observed sensor data
        Y = (self.snr * X @ np.diag(self.E) + N) @ self.F

        if self.nonlinear:
            Y = 1. / (1. + np.exp(-Y))

        # return inputs, outputs, and solution
        return scale(X), scale(Y)


def run(args=dict()):
    import warnings
    warnings.filterwarnings("ignore")

    models = {
        "Forward": Forward(),
        "Backward": Backward(),
        "PLS": GridPLS(),
        "CCA": GridCCA(),
        "RegCCA": GridRegCCA(),
        "B2B": B2B(),
    }

    n_samples = args.get('n_samples', 1000)
    dim_x = args.get('dim_x', 100)
    dim_y = args.get('dim_y', 100)
    snr = args.get('snr', 1)
    nc = args.get('nc', 5)
    nonlinear = args.get('nonlinear', 0)
    n_seeds = args.get('n_seeds', 10)
    refit_ko = args.get('refit_ko', False)

    results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        # Make environment
        synthetic = Synthetic(dim_x, dim_y, nc,
                              snr, nonlinear)

        # Make data
        X_train, Y_train = synthetic.sample(n_samples)
        X_test, Y_test = synthetic.sample(n_samples * 10)

        for m, model in models.items():

            # fit model on training data
            start = time.time()
            model.fit(X_train, Y_train)
            duration = time.time() - start

            # Estimate effect from model parameters
            auc = roc_auc_score(synthetic.E, model.E_)

            # Estimate effect from prediction reliability on held-out data
            XY_train = None if not refit_ko else (X_train, Y_train)
            r_delta = score_knockout(model, X_test, Y_test, XY_train)

            r_in = r_delta[synthetic.E == 1].mean()
            r_out = r_delta[synthetic.E == 0].mean()

            # Store results
            id = '_'.join(map(str, [dim_x, dim_y, nc, snr,
                                    nonlinear, m, seed]))
            result = dict(dim_x=dim_x, dim_y=dim_y, nc=nc, snr=snr,
                          nonlinear=nonlinear, model=m, seed=seed,
                          r_in=r_in, r_out=r_out, auc=auc, duration=duration,
                          id=id)
            print(result)
            results.append(result)
    return results

run()
