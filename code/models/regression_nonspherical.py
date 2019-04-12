# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence: BSD 3-clause

"""Implement regularized regression with non-spherical priors as described in:
   Nunez-Elizalde*, Huth* & Gallant (2016) Improving predictive models using
   non-spherical Gaussian priors, Society for Neuroscience (Poster).
"""
import numpy as np
from numpy.random import rand, randn
from scipy.linalg import inv, sqrtm
from sklearn.linear_model import Ridge


def r2(a, b):
    return np.corrcoef(a.squeeze(), b.squeeze())[0, 1]


def ridge(X, y, alpha=1.):
    """Classic ridge (i.e. spherical priors)"""
    Id = np.eye(X.shape[1])
    return inv(X.T.dot(X) + alpha * Id) @ X.T.dot(y)


def tikhonov_ns_prior(X, y, S, alpha=1.):
    """Tikhonov with non spherical gaussian priors"""
    S_sqrt = sqrtm(S)
    A = X.dot(S_sqrt)
    Id = np.eye(A.shape[1])
    beta_A = inv(A.T.dot(A) + alpha * Id) @ A.T.dot(y)
    return S_sqrt.dot(beta_A)


class NonSphericalRidge(Ridge):
    def __init__(self, S, alpha=1):
        # TODO add intercept
        self.S = S

    def fit(self, X, y):
        self.coef_ = tikhonov_ns_prior(X, y, self.S, self.alpha)
        return self


if __name__ == '__main__':
    r_ridge, r_ns_prior = list(), list()  # initialize scoring metrics

    for voxel in range(100):
        print(voxel)
        # Simulate data
        n_samples, n_features = 200, 500

        # Independent features
        X = randn(n_samples, n_features)

        # Make non-orthogonal features
        S = rand(n_features, n_features)
        S = S @ S.T  # ensure positive definite covariance

        X = X @ S.T

        # True coefficients
        beta = randn(n_features, 1)

        # Observables
        noise = randn(n_samples, 1)
        y = np.dot(X, beta) + noise

        # Estimate beta with or without spherical priors
        beta_hat = ridge(X, y, alpha=1.)
        beta_ns_hat = tikhonov_ns_prior(X, y, S, alpha=1.)

        # Compute correlation between estimates and ground truth
        r_ridge.append(r2(beta, beta_hat))
        r_ns_prior.append(r2(beta, beta_ns_hat))

    # print average scoring metrics
    for name, r in (('ridge', r_ridge), ('non spherical prior', r_ns_prior)):
        print('%s: %.2f (+/- %.2f)' % (
            name, np.mean(r, axis=0), np.std(r, axis=0)))
