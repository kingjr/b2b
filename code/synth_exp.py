from models import JRR, OLS, Ridge, Oracle, PLS, Lasso, RRR, CCA
from models_k import JRR_k, Ridge_k, CCA_k
from sklearn.metrics import roc_auc_score
from data import Synthetic
import numpy as np
import argparse
import random


models = {
    # "JRR": JRR,
    # "OLS": OLS,
    # "Ridge": Ridge,
    # "CCA": CCA,
    # "Oracle": Oracle,
    # "PLS": PLS,
    # "RRR": RRR,
    # "Lasso": Lasso,
    "JRR_k": JRR_k,
    "Ridge_k": Ridge_k,
    "CCA_k": CCA_k
}


def nmse(y_pred, y_true):
    num = np.power(np.linalg.norm(y_pred - y_true), 2)
    den = np.power(np.linalg.norm(y_true), 2)
    return num / den


def run_experiment(args):
    for seed in range(args.n_seeds):
        random.seed(seed)
        np.random.seed(seed)

        synthetic = Synthetic(args.dim_x,
                              args.dim_y,
                              args.nc,
                              args.snr,
                              args.nonlinear)

        # true binary E
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

            result = dict(vars(args))
            result["model"] = m
            result["seed"] = seed
            result["result_error_in_all"] = error_in_all
            result["result_error_out_all"] = error_out_all
            result["result_auc"] = roc_auc_score(mask_true, mask)
            results.append(result)

            print(results[-1])

    return str(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='B2BR synthetic experiment')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--dim_x', type=int, default=100)
    parser.add_argument('--dim_y', type=int, default=100)
    parser.add_argument('--snr', type=float, default=0.1)
    parser.add_argument('--nc', type=int, default=5)
    parser.add_argument('--nonlinear', type=int, default=0)
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    run_experiment(args)
