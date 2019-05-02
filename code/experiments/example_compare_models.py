import numpy as np
from sklearn.linear_model import RidgeCV
from models import JRR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
from base import make_data
from time import time


def norm_error(Y, Y_hat):
    return np.linalg.norm(Y - Y_hat) / np.linalg.norm(Y)


def evaluate(model, X, Y, E, name):
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
    diag_E = np.diag(E) > 0

    if 'jrr' in name:
        diag_E_hat = np.diag(coef)
    else:
        diag_E_hat = np.mean(coef**2, 1)

    auc = roc_auc_score(diag_E, diag_E_hat)

    score = dict(model=name,
                 train_err=train_err,
                 test_err=test_err,
                 auc=auc, duration=duration)
    return score, coef


if __name__ == '__main__':

    alphas = np.logspace(2, 2, 5)

    models = dict(
        ridge=RidgeCV(alphas=alphas),
        jrr=JRR(),
    )

    scores = list()
    for repeat in trange(100):
        X, Y, E, F, M = make_data(1000, nY=30, nX=25, nM=25, selected=.2,
                                  snr=.3, random_seed=repeat)

        # evaluate each model
        for idx, (name, model) in enumerate(models.items()):
            score, EF_hat = evaluate(model, X, Y, E, name)
            score['model'] = name
            score['seed'] = repeat
            scores.append(score)
    scores = pd.DataFrame(scores)

    plt.subplot(221)
    sns.boxplot(x='test_err', y='model', orient='h', data=scores)
    plt.axvline(1., color='k', linestyle=':')

    plt.subplot(223)
    sns.boxplot(x='auc', y='model', orient='h', data=scores)
    plt.axvline(1., color='k', linestyle=':')

    plt.subplot(122)
    plt.plot(range(2), range(2), label='chance', color='k', lw=1)
    plt.scatter(scores.loc[scores.model == 'jrr', 'auc'],
                scores.loc[scores.model == 'ridge', 'auc'],
                s=1)
    plt.scatter(scores.loc[scores.model == 'jrr', 'auc'].mean(),
                scores.loc[scores.model == 'ridge', 'auc'].mean(),
                s=100)
    plt.xlabel('jrr')
    plt.ylabel('ridge')
    plt.xlim(.5, 1)
    plt.ylim(.5, 1)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()
