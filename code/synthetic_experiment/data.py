import numpy as np
import math


class Synthetic(object):
    def __init__(self,             # number of samples
                 dim_x=50,         # number of features
                 dim_y=30,         # number of sensors
                 nc=5,             # number of selected features
                 snr=1.0,          # signal-to-noise ratio
                 nonlinear=False):  # number of selected features
        # linear transformation
        self.F = np.random.randn(dim_x, dim_y) / math.sqrt(dim_x)

        # masking transformation
        self.E = np.array([0] * (dim_x - nc) + [1] * (nc))
        self.E = np.diag(self.E)

        # features covariance
        self.cov_X1 = np.random.randn(dim_x, dim_x) / math.sqrt(dim_x)
        self.cov_X2 = np.random.randn(dim_x, dim_x) / math.sqrt(dim_x)

        # noise covariance
        self.cov_N1 = np.random.randn(dim_x, dim_x) / math.sqrt(dim_x)
        self.cov_N2 = np.random.randn(dim_x, dim_x) / math.sqrt(dim_x)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nonlinear = nonlinear
        self.snr = snr

    def sample(self,
               n_samples=1000,
               in_domain=True):
        if in_domain:
            X = np.random.randn(n_samples, self.dim_x) @ self.cov_X1
            N = np.random.randn(n_samples, self.dim_x) @ self.cov_N1
        else:
            X = np.random.randn(n_samples, self.dim_x) @ self.cov_X2
            N = np.random.randn(n_samples, self.dim_x) @ self.cov_N2

        # observed sensor data
        Y = (self.snr * X @ self.E + N) @ self.F

        if self.nonlinear:
            Y = 1. / (1. + np.exp(-Y))

        # return inputs, outputs, and solution
        return X, Y

    def solution(self):
        return np.diag(self.E)
