import numpy as np
from base import make_data
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV
from models import EstimatorCV, Encoding, JRR, ReducedRankRegressor
# from models import KernelCCA   # bug with gridcv?
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.cross_decomposition import PLSCanonical, PLSSVD  TODO
from sklearn.metrics import roc_auc_score
from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')

alphas = np.logspace(3, 3, 10)


models = dict(
    ols=LinearRegression(),
    ridge=RidgeCV(alphas=alphas),
    lasso=MultiTaskLassoCV(eps=1e2),  # default alphas better but slow
    pca_l2=EstimatorCV(Encoding(ols=RidgeCV()), dim='Y'),
    pca_l1=EstimatorCV(Encoding(ols=MultiTaskLassoCV(eps=1e2)), dim='Y'),
    cca=EstimatorCV(CCA(), dim='X'),
    # cca_l2=GridSearchCV(KernelCCA(kernel='linear'), {'reg': alphas}),  # bug
    pls=EstimatorCV(PLSRegression(), dim='X'),
    # TODO: pls_l2
    rrr=EstimatorCV(ReducedRankRegressor(), dim='X'),
    rrr_l2=GridSearchCV(ReducedRankRegressor(n_components=-1), {'reg': alphas}), # noqa
    jrr_nobagging=JRR(bagging=None),
    jrr_noreg=JRR(G=LinearRegression(), H=LinearRegression()),
    jrr_offdiag=JRR(zero_off_diag=False),
    jrr=JRR(),
)


def norm_error(Y, Y_hat):
    return np.linalg.norm(Y - Y_hat) / np.linalg.norm(Y)


def evaluate(model, X, Y, E):
    train, test = range(0, len(X), 2), range(1, len(X), 2)

    # Fit model
    start = time()
    model.fit(X[train], Y[train])
    duration = time() - start

    # Estimate fit and prediction error
    Y_hat = model.predict(X)
    train_err = norm_error(Y[train], Y_hat[train])
    test_err = norm_error(Y[test], Y_hat[test])

    # Retrieve fitted coeffs identification
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    if 'jrr' in name:
        coef = model.E_
    else:
        coef = model.coef_
    if name in ('ols', 'ridge', 'lasso'):
        coef = coef.T  # sklearn inconsistency

    # estimate whether E can be identified from coefs
    diag_E = np.sum(E**2, 0) > 0

    if 'jrr' in name:
        # JRR estimate E_hat: i.e. coefficient vary around 0
        diag_E_hat = np.diag(coef)
    else:
        # other methods estimate EF: i.e. loading needs to be
        # derived from absolute values
        diag_E_hat = np.mean(coef**2, 1)

    auc = roc_auc_score(diag_E, diag_E_hat)

    score = dict(model=name, snr=snr,
                 train_err=train_err,
                 test_err=test_err,
                 auc=auc,
                 duration=duration)
    return score, coef


if __name__ == '__main__':
    # Plot EF estimates on an example data with two levels of
    # signal-to-noise ratios

    for snr in (.5, .3):
        E = np.diag(np.r_[np.zeros(20), np.ones(5)])
        X, Y, E, F = make_data(nY=30, nX=30, selected=.2, snr=snr, E=E,
                               random_seed=0)

        EF = E @ F.T
        vmax = EF.max()
        fig, axes = plt.subplots(1, 2, sharey=True)
        axes[0].matshow(EF, cmap='RdBu_r')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('EF')
        axes[1].plot(np.mean(E**2, 1), range(len(E)))
        axes[1].set_yticks([])

        # Derive E or EF from each model
        for idx, (name, model) in enumerate(models.items()):
            score, EF_hat = evaluate(model, X, Y, E)

            fig, axes = plt.subplots(1, 2, sharey=True)

            vmax = EF_hat.max()
            axes[0].matshow(EF_hat, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            title = '%s\n auc=%.2f; err=%.3f'
            axes[0].set_title(title % (name, score['auc'], score['test_err']))
            axes[1].plot(np.mean(EF_hat**2, 1), range(len(E)))
            axes[1].set_yticks([])

    # run over multiple 50 examples to estimate robustness.
    # This should be re-run properly with larger nested grid search to ensure a
    # fair comparison between models.
    scores = list()
    for repeat in trange(50):
        snr = .3
        E = np.diag(np.r_[np.zeros(20), np.ones(5)])
        X, Y, E, F = make_data(nY=30, nX=30, selected=.2, snr=snr, E=E,
                               random_seed=repeat)

        # evaluate each model
        for idx, (name, model) in enumerate(models.items()):
            score, EF_hat = evaluate(model, X, Y, E)
            score['snr'] = snr
            score['model'] = name
            scores.append(score)
    scores = pd.DataFrame(scores)

    fig, axes = plt.subplots(1, 4, sharey=True, figsize=[12, 4])
    sns.boxplot(x='train_err', y='model', orient='h', data=scores, ax=axes[0])
    axes[0].axvline(1., color='k', linestyle=':')

    sns.boxplot(x='test_err', y='model', orient='h', data=scores, ax=axes[1])
    axes[1].axvline(1., color='k', linestyle=':')

    sns.boxplot(x='auc', y='model', orient='h', data=scores, ax=axes[2])
    axes[2].axvline(.5, color='k', linestyle=':')

    sns.boxplot(x='duration', y='model', orient='h', data=scores, ax=axes[3])
    plt.xscale('log')
