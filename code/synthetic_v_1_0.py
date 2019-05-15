import numpy as np
from scipy.linalg import sqrtm


def rolling_covariance(rho, dim):
    cov = np.zeros((dim, dim))
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            cov[i, j] = np.power(rho, np.abs(i - j) / dim)
    return cov


def synthetic_data_v_1_0(n_samples=1000,   # number of samples
                         dim_x=50,         # number of features
                         rho_x=0.5,        # correlation of features
                         rho_n=0.5,        # correlation of noise
                         dim_y=30,         # number of sensors
                         snr=1.0,          # signal-to-noise ratio
                         nc=5,             # number of selected features
                         nonlinear=False):  # number of selected features
    # linear transformation
    F = np.random.randn(dim_x, dim_y) / np.sqrt(dim_x)

    # masking transformation
    E = np.array([0] * (dim_x - nc) + [1] * (nc))
    np.random.shuffle(E)
    E = np.diag(E)

    # features
    cov_X = rolling_covariance(rho_x, dim_x)
    X = np.random.randn(n_samples, dim_x) @ sqrtm(cov_X)
    np.random.shuffle(X.T)

    # noise
    cov_N = rolling_covariance(rho_n, dim_x)
    N = np.random.randn(n_samples, dim_x) @ sqrtm(cov_N)
    np.random.shuffle(N.T)

    # observed sensor data
    Y = (snr * X @ E + N) @ F

    if nonlinear:
        Y = 1. / (1. + math.exp(-Y))

    # return inputs, outputs, and solution
    return X, Y, E


sweep_v_1_0 = {
    "n_samples": [100, 1000, 10000],
    "rho_x": [0, 0.01, 0.2, 0.95],
    "rho_n": [0, 0.01, 0.2, 0.95],
    "dim_x": [10, 50, 100, 500],
    "dim_y": [10, 50, 100, 500],
    "snr": [0.1, .25, 1],
    "nc": [1, 10, 20],
    "nonlinear": [0, 1],
}
