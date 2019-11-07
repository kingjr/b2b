import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from regression import ridge_cv, r_score, fit_knockout
from base import rn_score


class B2B():
    def __init__(self, alphas=np.logspace(-4, 4, 20),
                 independent_alphas=True, ensemble=None):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.ensemble = ensemble

    def fit(self, X, Y):

        self.G_ = list()
        self.H_ = list()
        self.Ks_ = list()

        # Prepare ensembling
        if self.ensemble is None:
            ensemble = [(range(len(X)), range(len(X))), ]
        else:
            if isinstance(self.ensemble, int):
                ensemble = ShuffleSplit(self.ensemble)
            else:
                ensemble = self.ensemble
            ensemble = [split for split in ensemble.split(X)]

        # Ensembling loop
        for train, test in ensemble:

            # Fit decoder
            G, G_alpha, YG = ridge_cv(Y[train], X[train],
                                      self.alphas,
                                      self.independent_alphas)
            self.G_.append(G)

            if len(X[train]) != len(X):
                YG = Y @ G.T

            # Fit encoder
            H, H_alpha, _ = ridge_cv(X[test], YG[test],
                                     self.alphas,
                                     self.independent_alphas)
            self.H_.append(H)

            # Fit knock-out encoders
            Ks = fit_knockout(X, YG,
                              self.alphas,
                              self.independent_alphas, True)
            self.Ks_.append(Ks)

        # Aggregate ensembling
        self.G_ = np.mean(self.G_, 0)
        self.H_ = np.mean(self.H_, 0)
        self.Ks_ = np.mean(self.Ks_, 0)

        self.E_ = np.diag(self.H_)

        return self

    def score(self, X, Y, scoring='r', multioutput='raw'):
        # Transform with decoder
        YG = Y @ self.G_.T
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column X
        return rn_score(YG, XH, scoring=scoring, multioutput=multioutput)

    def score_knockout(self, X, Y, scoring='r', multioutput=None):
        # Transform with decoder
        YG = Y @ self.G_.T

        # For each knock-out, compute R score
        K_scores = list()
        for xi, K in enumerate(self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            K_scores.append(rn_score(YG[:, xi], XK[:, xi],
                                     scoring=scoring, multioutput='raw')[0])

        return np.array(K_scores)
