from models import JRR, OLS, Ridge, Oracle, PLS, Lasso
from data import Synthetic

import numpy as np
import argparse
import random


def nmse(y_pred, y_true):
    num = np.power(np.linalg.norm(y_pred - y_true), 2)
    den = np.power(np.linalg.norm(y_true), 2)
    return num / den


def compute_false_positives(e_pred, e_true):
    return np.mean(((e_pred == 1) * (e_true == 0) * 1.0))


def compute_false_negatives(e_pred, e_true):
    return np.mean(((e_pred == 0) * (e_true == 1) * 1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JRR synthetic experiment')
    parser.add_argument('--model', type=str, default='lasso')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--dim_x', type=int, default=50)
    parser.add_argument('--dim_y', type=int, default=30)
    parser.add_argument('--rho_x', type=float, default=0.5)
    parser.add_argument('--rho_n', type=float, default=0.5)
    parser.add_argument('--snr', type=float, default=0.1)
    parser.add_argument('--nc', type=int, default=5)
    parser.add_argument('--nonlinear', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    synthetic = Synthetic(args.dim_x,
                          args.dim_y,
                          args.rho_x,
                          args.rho_n,
                          args.nc,
                          args.snr,
                          args.nonlinear)

    # training data
    x_tr, y_tr = synthetic.sample(args.n_samples)

    # testing data (in-domain)
    x_te_in, y_te_in = synthetic.sample(args.n_samples * 10)

    # testing data (out-domain)
    x_te_out, y_te_out = synthetic.sample(args.n_samples * 10, in_domain=False)

    if args.model == "ols":
        model = OLS()
    elif args.model == "ridge":
        model = Ridge()
    elif args.model == "oracle":
        model = Oracle(synthetic.solution())
    elif args.model == "jrr":
        model = JRR() 
    elif args.model == "pls":
        model = PLS() 
    elif args.model == "lasso":
        model = Lasso() 
    else:
        raise NotImplementedError

    # fit model on training data
    model.fit(x_tr, y_tr)

    error_in_all = nmse(model.predict(x_te_in), y_te_in)
    error_out_all = nmse(model.predict(x_te_out), y_te_out)

    # this is the binary mask E estimated by the model
    mask = model.solution()

    false_positives = compute_false_positives(mask, synthetic.solution())
    false_negatives = compute_false_negatives(mask, synthetic.solution())

    # fit a basic model on the selected causes
    selected = np.nonzero(mask)[0]

    model = OLS()
    model.fit(x_tr[:, selected], y_tr)

    error_in_mask = nmse(model.predict(x_te_in[:, selected]), y_te_in)
    error_out_mask = nmse(model.predict(x_te_out[:, selected]), y_te_out)

    result = dict(vars(args))
    result["result_error_in_all"] = error_in_all
    result["result_error_out_all"] = error_out_all
    result["result_error_in_mask"] = error_in_mask
    result["result_error_out_mask"] = error_out_mask
    result["result_false_positives"] = false_positives
    result["result_false_negatives"] = false_negatives

    print(result)
