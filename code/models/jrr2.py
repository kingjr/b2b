import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit


def sonquist_morgan(x):
    z=np.sort(x)
    n=z.size
    m1=0
    m2=np.sum(z)
    mx=0
    best=-1
    for i in range(n-1): 
        m1+=z[i]
        m2-=z[i]
        ind=(i+1)*(n-i-1)*(m1/(i+1)-m2/(n-i-1))**2
        if ind>mx :
            mx=ind
            best=z[i]
    K=mx/(np.var(x)*n) #significance indicator, not used for now
    return 1*(x>(best+1.0e-6))


class JRR(object):
    """Finds E in:
        Y = F(EX+N)

    Where Y is the recordings (samples * sensors),
          F is an unknown forward operator (neurons * sensors),
          X is the factors (samples * sources),
          N is a source noise (samples * sources)

    Step 1: Bagging
    for set1, set2 in partitions:
        Step 1.1: G = ls(Y_1, X_1)
        Step 1.2: H = ls(X_2, Y_2 * G)
    Step 2 (optional): E = diag(H)
    Step 3: F = ls(XE, Y)
    """

    def __init__(self, G=None, H=None, F=None, bagging=10, zero_off_diag=True):
        self.G = RidgeCV() if G is None else G
        self.H = RidgeCV() if H is None else H
        self.F = RidgeCV() if F is None else F
        self.bagging = bagging
        self.zero_off_diag = zero_off_diag

    def fit(self, X, Y):
        if self.bagging in (0, False, None):
            Gset = range(len(X))
            Hset = range(len(X))
            ensemble = [(Gset, Hset)]
        else:
            bagging = ShuffleSplit(self.bagging, test_size=.5)
            ensemble = [split for split in bagging.split(X, Y)]

        H = list()
        for train, test in ensemble:
            X_hat = self.G.fit(Y[train], X[train]).predict(Y)
            H += [self.H.fit(X[test], X_hat[test]).coef_, ]

        # Estimate E
        self.E_ = np.mean(H, 0).T  # TODO: check transpose
        if self.zero_off_diag:
            #self.E_ = np.diag(np.diag(self.E_))
            self.E_ = np.diag(sonquist_morgan(np.diag(self.E_)))
        # Filter X by E hat and estimate F
        self.F.fit(X @ self.E_, Y)
        return self

    def predict(self, X):
        return self.F.predict(X @ self.E_)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
    from sklearn.preprocessing import scale
    from sklearn.metrics import roc_auc_score
    # Simulate data
    """Y = F(EX+N)"""

    np.random.seed(0)

    # Problem dimensionality
    n = 1000
    nE = nX = 10
    nY = 10
    snr = .25  # signal to noise ratio
    selected = .5  # number of X feature selected by E

    selected = min(int(np.floor(selected*nX)) + 1, nX-1)
    E = np.identity(nX)
    E[selected:] = 0

    # X covariance
    Cx = np.random.randn(nX, nX)
    Cx = Cx.dot(Cx.T) / nX  # sym pos-semidefin
    X = np.random.multivariate_normal(np.zeros(nX), Cx, n)

    # Noise (homosedastic in source space)
    N = np.random.randn(n, nE)

    # Forward operator (linear mixture)
    F = np.random.randn(nY, nE)

    Y = ((X @ E.T) * snr + N) @ F.T

    X = scale(X)
    Y = scale(Y)

    # Fit method
    jrr = JRR()
    train, test = range(0, n, 2), range(1, n, 2)
    E_hat = jrr.fit(X[train], Y[train]).E_
    score = jrr.score(X[test], Y[test])

    #compare extraction to E
    extraction_score=np.sum(abs(np.diag(E)-np.diag(E_hat))

    #WARNING: this might change now that E_at has been replaced by the sonquist version...
    # it should make us better, as we don't care anymore about the order of diagonal coeffs, just their class
    print('E_auc', roc_auc_score(np.diag(E), np.diag(E_hat)))
    print('Y_score', score)
