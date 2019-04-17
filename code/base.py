import numpy as np
from sklearn.preprocessing import scale


def make_data(n: int = 1000,  # number of samples
              nX: int = 100,  # dimensionality of X
              nE: int = 10,  # dimensionality of E
              nY: int = 10,  # dimensionality of Y
              random_seed: int = None,
              snr: float = 1e-1,  # signal to noise ratio
              selected: float = .5,  # proportion of selected feature in E
              E: np.ndarray = None,  # selecting matrix
              F: np.ndarray = None,  # mixing matrix
              Cx: np.ndarray = None,  # feature covariance
              heteroscedastic: bool = False,  # noise type
              ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    """Y = F(EX+N*snr)
    returns X, Y, E, F
    """
    np.random.seed(random_seed)

    if E is None:
        E = np.zeros((nE, nX))
        # selected must be between 1 and nX-1 to ensure that there is at least
        # one selected and one unselected X feature
        selected = min(int(np.floor(selected*nX)) + 1, nX-1)
        E[:, :selected] = np.random.randn(nE, selected)
    else:
        nE, nX = E.shape

    if Cx is None:
        Cx = np.random.randn(nX, nX)
        Cx = Cx.dot(Cx.T) / nX  # sym pos-semidefin
    else:
        nX = len(Cx)
    X = np.random.multivariate_normal(np.zeros(nX), Cx, n)

    N = np.random.randn(n, nE)
    if heteroscedastic:
        Cn = np.random.randn(nE, nE)
        Cn = Cn.dot(Cn.T) / nE  # sym pos-semidefin
        N = np.random.multivariate_normal(np.zeros(nE), Cn, n)

    if F is None:
        F = np.random.randn(nY, nE)
    else:
        nY = len(F)

    S = (X @ E.T) * snr
    Y = (S+N) @ F.T

    X = scale(X)
    Y = scale(Y)

    return X, Y, E, F
