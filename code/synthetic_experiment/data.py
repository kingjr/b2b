import numpy as np
from scipy.linalg import sqrtm


def rolling_covariance(rho, dim):
    cov = np.zeros((dim, dim))
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            cov[i, j] = np.power(rho, np.abs(i - j) / dim)
    return cov


class Synthetic(object):
    def __init__(self,            # number of samples
                dim_x=50,         # number of features
                dim_y=30,         # number of sensors
                rho_x=0.5,        # correlation of features
                rho_n=0.5,        # correlation of noise
                nc=5,             # number of selected features
                snr=1.0,          # signal-to-noise ratio
                nonlinear=False): # number of selected features
        # linear transformation
        self.F = np.random.randn(dim_x, dim_y) / np.sqrt(dim_x)

        # masking transformation
        self.E = np.array([0] * (dim_x - nc) + [1] * (nc))
        np.random.shuffle(self.E)
        self.E = np.diag(self.E)

        # features covariance
        self.cov_X = rolling_covariance(rho_x, dim_x)
        self.cols_X = np.random.permutation(dim_x)

        # noise covariance
        self.cov_N = rolling_covariance(rho_n, dim_x)
        self.cols_N = np.random.permutation(dim_x)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nonlinear = nonlinear
        self.snr = snr

    def sample(self,
               n_samples=1000,
               in_domain=True):
        X = np.random.randn(n_samples, self.dim_x)
        if in_domain:
            X = X @ sqrtm(self.cov_X)
        X = X[:, self.cols_X]

        N = np.random.randn(n_samples, self.dim_x)
        if in_domain:
            N = N @ sqrtm(self.cov_N)
        N = N[:, self.cols_N]

        # observed sensor data
        Y = (self.snr * X @ self.E + N) @ self.F

        if self.nonlinear:
            Y = 1. / (1. + math.exp(-Y))

        # return inputs, outputs, and solution
        return X, Y

    def solution(self):
        return np.diag(self.E)
