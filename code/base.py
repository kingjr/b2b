import numpy as np
from sklearn.preprocessing import scale


def make_data(n: int = 1000,  # number of samples
              nX: int = 100,  # dimensionality of X
              nY: int = 10,  # dimensionality of Y
              nM: int = 10,  # dimensionality of source
              random_seed: int = None,
              snr: float = 1e-1,  # signal to noise ratio
              selected: float = .5,  # proportion of selected feature in E
              E: np.ndarray = None,  # selecting matrix
              M: np.ndarray = None,  # selecting matrix
              F: np.ndarray = None,  # mixing matrix
              Cx: np.ndarray = None,  # feature covariance
              heteroscedastic: bool = False,  # noise type
              ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    """Y = F(MEX+N*snr)
    returns X, Y, E, F
    """
    np.random.seed(random_seed)

    if E is None:
        E = np.zeros(nX)
        # selected must be between 1 and nX-1 to ensure that there is at least
        # one selected and one unselected X feature
        selected = min(int(np.floor(selected*nX)) + 1, nX-1)
        E[:selected] = 1
        E = np.diag(E)
    else:
        nX, nX = E.shape

    if M is None:
        M = np.random.randn(nM, nX)
        M /= np.sqrt(np.sum(M**2, 1, keepdims=True))
    else:
        nM = len(M)

    if Cx is None:
        Cx = np.random.randn(nX, nX)
        Cx = Cx.dot(Cx.T) / nX  # sym pos-semidefin
    else:
        nX = len(Cx)
    X = np.random.multivariate_normal(np.zeros(nX), Cx, n)

    N = np.random.randn(n, nM)
    if heteroscedastic:
        Cn = np.random.randn(nM, nM)
        Cn = Cn.dot(Cn.T) / nX  # sym pos-semidefin
        N = np.random.multivariate_normal(np.zeros(nX), Cn, n)

    if F is None:
        F = np.random.randn(nY, nM)
    else:
        nY = len(F)

    Y = ((X @ E.T @ M.T) * snr + N) @ F.T

    X = scale(X)
    Y = scale(Y)

    return X, Y, E, F
