# TODO: model Y = (XE + N)F + N' with second noise N'
# TODO: uniform distributions

from models import JRR, OLS, Ridge, Oracle, PLS, Lasso, RRR, CCA
from sklearn.metrics import roc_auc_score
from data import Synthetic

import numpy as np
import argparse
import random

models = {
    "OLS": OLS,
    "Ridge": Ridge,
    "CCA": CCA,
    "Oracle": Oracle,
    "JRR": JRR,
    "PLS": PLS,
    "RRR": RRR,
    "Lasso": Lasso
}


def sonquist_morgan(x):
    z = np.sort(x)
    n = z.size
    m1 = 0
    m2 = np.sum(z)
    mx = 0
    best = -1
    for i in range(n - 1):
        m1 += z[i]
        m2 -= z[i]
        ind = (i + 1) * (n - i - 1) * (m1 / (i + 1) - m2 / (n - i - 1))**2
        if ind > mx:
            mx = ind
            best = z[i]
    res = [0 for i in range(n)]
    for i in range(n):
        if x[i] > best:
            res[i] = 1
    return np.array(res)


def nmse(y_pred, y_true):
    num = np.power(np.linalg.norm(y_pred - y_true), 2)
    den = np.power(np.linalg.norm(y_true), 2)
    return num / den


def compute_false_positives(e_pred, e_true):
    return np.mean(((e_pred == 1) * (e_true == 0) * 1.0))


def compute_false_negatives(e_pred, e_true):
    return np.mean(((e_pred == 0) * (e_true == 1) * 1.0))


def run_experiment(args):
    for seed in range(args.n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        synthetic = Synthetic(args.dim_x,
                              args.dim_y,
                              args.nc,
                              args.snr,
                              args.nonlinear)

        mask_true = synthetic.solution()

        # training data
        x_tr, y_tr = synthetic.sample(args.n_samples)

        # testing data (in-domain)
        x_te_in, y_te_in = synthetic.sample(args.n_samples * 10)

        # testing data (out-domain)
        x_te_out, y_te_out = synthetic.sample(args.n_samples * 10,
                                              in_domain=False)

        results = []

        for m, Model in models.items():
            if m == "Oracle":
                model = Model(mask_true)
            else:
                model = Model()

            # fit model on training data
            model.fit(x_tr, y_tr)

            error_in_all = nmse(model.predict(x_te_in), y_te_in)
            error_out_all = nmse(model.predict(x_te_out), y_te_out)

            # this is the binary mask E estimated by the model
            mask = model.solution()

            false_positives = compute_false_positives(mask, mask_true)
            false_negatives = compute_false_negatives(mask, mask_true)

            # fit a basic model on the selected causes
            sel = np.nonzero(sonquist_morgan(mask))[0]

            model = Ridge()
            model.fit(x_tr[:, sel], y_tr)

            error_in_mask = nmse(model.predict(x_te_in[:, sel]), y_te_in)
            error_out_mask = nmse(model.predict(x_te_out[:, sel]), y_te_out)

            result = dict(vars(args))
            result["model"] = m
            result["seed"] = seed
            result["result_error_in_all"] = error_in_all
            result["result_error_out_all"] = error_out_all
            result["result_error_in_mask"] = error_in_mask
            result["result_error_out_mask"] = error_out_mask
            result["result_false_positives"] = false_positives
            result["result_false_negatives"] = false_negatives
            result["result_auc"] = roc_auc_score(mask_true, mask) 
            results.append(result)

            print(results[-1])

    return str(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JRR synthetic experiment')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--dim_x', type=int, default=100)
    parser.add_argument('--dim_y', type=int, default=100)
    parser.add_argument('--snr', type=float, default=0.1)
    parser.add_argument('--nc', type=int, default=5)
    parser.add_argument('--nonlinear', type=int, default=0)
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    run_experiment(args)
