"""Python module for regularized kernel canonical correlation analysis

Slightly Adapted from https://github.com/gallantlab/pyrcca

Licence
Copyright (c) 2015, The Regents of the University of California (Regents).
All rights reserved.

Permission to use, copy, modify, and distribute this software and its
documentation for educational, research, and not-for-profit purposes, without
fee and without a signed licensing agreement, is hereby granted, provided that
the above copyright notice, this paragraph and the following two paragraphs
appear in all copies, modifications, and distributions. Contact The Office of
Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley,
CA 94720-1620, (510) 643-7201, for commercial licensing opportunities.

Created by Natalia Bilenko, University of California, Berkeley.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. REGENTS SPECIFICALLY DISCLAIMS ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND
ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS".
REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.
"""

# TODO: fix kernel prediction: currently only project to Z space

import joblib
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator


__copyright__ = 'Copyright 2016, UC Berkeley, Gallant lab.'


class _CCABase(object):
    def __init__(self, numCV=None, reg=None, regs=None, numCC=None,
                 numCCs=None, kernelcca=True, ktype=None, verbose=False,
                 select=0.2, cutoff=1e-15, gausigma=1.0, degree=2):
        self.numCV = numCV
        self.reg = reg
        self.regs = regs
        self.numCC = numCC
        self.numCCs = numCCs
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.select = select
        self.gausigma = gausigma
        self.degree = degree
        if self.kernelcca and self.ktype is None:
            self.ktype = 'linear'
        self.verbose = verbose

    def train(self, data):
        if self.verbose:
            print('Training CCA, kernel = %s, regularization = %0.4f, '
                  '%d components' % (self.ktype, self.reg, self.numCC))

        comps = compute_kcca(data, self.reg, self.numCC,
                             kernelcca=self.kernelcca,
                             ktype=self.ktype, gausigma=self.gausigma,
                             degree=self.degree)
        self.cancorrs, self.ws, self.comps = recon(data, comps,
                                                   kernelcca=self.kernelcca)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self

    def validate(self, vdata):
        vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        if not hasattr(self, 'ws'):
            raise NameError('Algorithm has not been trained.')
        self.preds, self.corrs = predict(vdata, self.ws, self.cutoff)
        return self.corrs

    def compute_ev(self, vdata):
        nD = len(vdata)
        nC = self.ws[0].shape[1]
        nF = [d.shape[1] for d in vdata]
        self.ev = [np.zeros((nC, f)) for f in nF]
        for cc in range(nC):
            ccs = cc+1
            if self.verbose:
                print('Computing explained variance for component #%d' % ccs)
            preds, corrs = predict(vdata, [w[:, ccs-1:ccs] for w in self.ws],
                                   self.cutoff)
            resids = [abs(d[0]-d[1]) for d in zip(vdata, preds)]
            for s in range(nD):
                ev = abs(vdata[s].var(0) - resids[s].var(0))/vdata[s].var(0)
                ev[np.isnan(ev)] = 0.
                self.ev[s][cc] = ev
        return self.ev


class CCACrossValidate(_CCABase):
    """
    Attributes:
        numCV (int): number of cross-validation folds
        regs (list or numpy.array): regularization param array.
                                   Default: np.logspace(-3, 1, 10)
        numCCs (list or numpy.array): list of numbers of canonical dimensions
                                     to keep. Default is np.range(5, 10).
        kernelcca (bool): kernel or non-kernel CCA. Default is True.
        ktype (string): type of kernel used if kernelcca is True.
                        Value can be 'linear' (default) or 'gaussian'.
        verbose (bool): default is True.

    Returns:
        ws (list): canonical weights
        comps (list): canonical components
        cancorrs (list): correlations of the canonical components
                         on the training dataset
        corrs (list): correlations on the validation dataset
        preds (list): predictions on the validation dataset
        ev (list): explained variance for each canonical dimension
    """

    def __init__(self, numCV=None, regs=None, numCCs=None, kernelcca=True,
                 ktype=None, verbose=False, select=0.2, cutoff=1e-15,
                 gausigma=1.0, degree=2):
        numCV = 10 if numCV is None else numCV
        regs = np.array(np.logspace(-3, 1, 10)) if regs is None else regs
        numCCs = np.arange(5, 10) if numCCs is None else numCCs
        super(CCACrossValidate, self).__init__(numCV=numCV, regs=regs,
                                               numCCs=numCCs,
                                               kernelcca=kernelcca,
                                               ktype=ktype, verbose=verbose,
                                               select=select, cutoff=cutoff,
                                               gausigma=gausigma,
                                               degree=degree)

    def train(self, data, parallel=True):
        """
        Train CCA with cross-validation for a set of regularization
        coefficients and/or numbers of CCs
        Attributes:
            data (list): training data matrices
                         (number of samples X number of features).
                         Number of samples must match across datasets.
            parallel (bool): use joblib to train cross-validation folds
                             in parallel
        """
        corr_mat = np.zeros((len(self.regs), len(self.numCCs)))
        selection = max(int(self.select * min([d.shape[1] for d in data])), 1)
        for ri, reg in enumerate(self.regs):
            for ci, numCC in enumerate(self.numCCs):
                running_corr_mean_sum = 0.
                if parallel:
                    fold_corr_means = joblib.Parallel(n_jobs=self.numCV)(
                        joblib.delayed(train_cvfold)(
                            data=data, reg=reg, numCC=numCC,
                            kernelcca=self.kernelcca, ktype=self.ktype,
                            gausigma=self.gausigma, degree=self.degree,
                            cutoff=self.cutoff, selection=selection)
                        for fold in range(self.numCV))
                    running_corr_mean_sum += sum(fold_corr_means)
                else:
                    for cvfold in range(self.numCV):
                        fold_corr_mean = train_cvfold(
                            data=data, reg=reg, numCC=numCC,
                            kernelcca=self.kernelcca, ktype=self.ktype,
                            gausigma=self.gausigma, degree=self.degree,
                            cutoff=self.cutoff, selection=selection)
                        running_corr_mean_sum += fold_corr_mean

                corr_mat[ri, ci] = running_corr_mean_sum / self.numCV
        best_ri, best_ci = np.where(corr_mat == corr_mat.max())
        self.best_reg = self.regs[best_ri[0]]
        self.best_numCC = self.numCCs[best_ci[0]]

        comps = compute_kcca(data, self.best_reg, self.best_numCC,
                             kernelcca=self.kernelcca, ktype=self.ktype,
                             gausigma=self.gausigma, degree=self.degree)
        self.cancorrs, self.ws, self.comps = recon(data, comps,
                                                   kernelcca=self.kernelcca)
        if len(data) == 2:
            self.cancorrs = self.cancorrs[np.nonzero(self.cancorrs)]
        return self


def train_cvfold(data, reg, numCC, kernelcca, ktype, gausigma, degree,
                 cutoff, selection):
    """
    Train a cross-validation fold of CCA
    """
    nT = data[0].shape[0]
    chunklen = 10 if nT > 50 else 1
    nchunks = int(0.2 * nT / chunklen)
    indchunks = zip(*[iter(range(nT))] * chunklen)
    np.random.shuffle(indchunks)
    heldinds = [ind for chunk in indchunks[:nchunks]
                for ind in chunk]
    notheldinds = list(set(range(nT)) - set(heldinds))
    comps = compute_kcca([d[notheldinds] for d in data], reg, numCC,
                         kernelcca=kernelcca, ktype=ktype,
                         gausigma=gausigma, degree=degree)
    cancorrs, ws, ccomps = recon([d[notheldinds] for d in data], comps,
                                 kernelcca=kernelcca)
    preds, corrs = predict([d[heldinds] for d in data], ws, cutoff=cutoff)
    fold_corr_mean = []
    for corr in corrs:
        corr_idx = np.argsort(corr)[::-1]
        corr_mean = corr[corr_idx][:selection].mean()
        fold_corr_mean.append(corr_mean)
    return np.mean(fold_corr_mean)


class CCA(_CCABase):
    """Attributes:
        reg (float): regularization parameter. Default is 0.1.
        numCC (int): number of canonical dimensions to keep. Default is 10.
        kernelcca (bool): kernel or non-kernel CCA. Default is True.
        ktype (string): type of kernel used if kernelcca is True.
                        Value can be 'linear' (default) or 'gaussian'.
        verbose (bool): default is True.

    Returns:
        ws (list): canonical weights
        comps (list): canonical components
        cancorrs (list): correlations of the canonical components
                         on the training dataset
        corrs (list): correlations on the validation dataset
        preds (list): predictions on the validation dataset
        ev (list): explained variance for each canonical dimension
    """
    def __init__(self, reg=0., numCC=10, kernelcca=True, ktype=None,
                 verbose=False, cutoff=1e-15):
        super(CCA, self).__init__(reg=reg, numCC=numCC, kernelcca=kernelcca,
                                  ktype=ktype, verbose=verbose, cutoff=cutoff)

    def train(self, data):
        return super(CCA, self).train(data)


def predict(vdata, ws, cutoff=1e-15):
    """Get predictions for each dataset based on the other datasets
    and weights. Find correlations with actual dataset."""
    iws = [np.linalg.pinv(w.T, rcond=cutoff) for w in ws]
    ccomp = _listdot([d.T for d in vdata], ws)
    ccomp = np.array(ccomp)
    preds = []
    corrs = []

    for dnum in range(len(vdata)):
        idx = np.ones((len(vdata),))
        idx[dnum] = False
        proj = ccomp[idx > 0].mean(0)
        pred = np.dot(iws[dnum], proj.T).T
        pred = np.nan_to_num(_zscore(pred))
        preds.append(pred)
        cs = np.nan_to_num(_rowcorr(vdata[dnum].T, pred.T))
        corrs.append(cs)
    return preds, corrs


def compute_kcca(data, reg=0., numCC=None, kernelcca=True, ktype='linear',
                 gausigma=1.0, degree=2):
    """Set up and solve the kernel CCA eigenproblem
    """
    if kernelcca:
        kernel = [_make_kernel(d, ktype=ktype, gausigma=gausigma,
                               degree=degree) for d in data]
    else:
        kernel = [d.T for d in data]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

    # Get the auto- and cross-covariance matrices
    crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH):
    LH = np.zeros((sum(nFs), sum(nFs)))
    RH = np.zeros((sum(nFs), sum(nFs)))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(nDs):
        RH[sum(nFs[:i]): sum(nFs[:i+1]),
           sum(nFs[:i]): sum(nFs[:i+1])] = (crosscovs[i * (nDs + 1)]
                                            + reg * np.eye(nFs[i]))

        for j in range(nDs):
            if i != j:
                LH[sum(nFs[:j]): sum(nFs[:j+1]),
                   sum(nFs[:i]): sum(nFs[:i+1])] = crosscovs[nDs * j + i]

    LH = (LH + LH.T) / 2.
    RH = (RH + RH.T) / 2.

    maxCC = LH.shape[0]
    r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]):sum(nFs[:i + 1]), :numCC])
    return comp


def recon(data, comp, corronly=False, kernelcca=True):
    # nT = data[0].shape[0]
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp


def _zscore(d): return (d - d.mean(0)) / d.std(0)


def _demean(d): return d - d.mean(0)


def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]


def _listcorr(a):
    """Returns pairwise row correlations for all items in array as
    a list of matrices
    """
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0, 1])
                                  for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs


def _rowcorr(a, b):
    """Correlations between corresponding matrix rows"""
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0, 1]
    return cs


def _make_kernel(d, normalize=True, ktype='linear', gausigma=1.0, degree=2):
    """Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree
    """
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == 'linear':
        kernel = np.dot(cd, cd.T)
    elif ktype == 'gaussian':
        from scipy.spatial.distance import pdist, squareform
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)
    elif ktype == 'poly':
        kernel = np.dot(cd, cd.T) ** degree
    elif not isinstance(ktype, np.ndarray):
        raise ValueError('unknown kernel')
    else:
        kernel = ktype
    kernel = (kernel + kernel.T) / 2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel


class KernelCCA(BaseEstimator):
    """Object to make pyrcca have sklearn API"""
    def __init__(self, reg=0, n_components=-1, kernel='linear'):
        if kernel == 'linear':
            self.cca = CCA(reg=reg, kernelcca=False,
                           numCC=n_components)
        else:
            kernel_ = kernel
            if kernel_ == 'rbf':  # to fit sklearn naming
                kernel_ = 'gaussian'
            self.cca = CCA(reg=reg, ktype=kernel_, kernelcca=True,
                           numCC=n_components)
        self.kernel = kernel
        self.n_components = n_components

    def fit(self, X, Y):
        self.x_scaler_ = StandardScaler()
        X = self.x_scaler_.fit_transform(X)
        self.y_scaler_ = StandardScaler()
        Y = self.y_scaler_.fit_transform(Y)
        if self.n_components == -1:
            self.cca.numCC = X.shape[1]
        self.cca.train([X, Y])
        wx, wy = self.cca.ws
        self.x_weights_ = wx
        self.y_weights_ = wy
        iwy = np.linalg.pinv(wy, rcond=self.cca.cutoff)
        self.coef_ = iwy.T @ wx.T
        self.coef_ = self.coef_.T  # to match sklearn CCA
        # TODO: find var mapping with sklearn
        return self

    def predict(self, X):
        if self.kernel != 'linear':
            print('kernel prediction are probably not valid')
        X = self.x_scaler_.transform(X)
        Y_hat = (self.coef_.T @ X.T).T
        Y_hat = self.y_scaler_.inverse_transform(Y_hat)
        return Y_hat

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
    # Initialize number of samples
    n_samples = 1000

    # Define two latent variables (number of samples x 1)
    L1, L2 = np.random.randn(2, n_samples,)

    # Define independent components for each dataset
    # (number of observations x dataset dimensions)
    indep1 = np.random.randn(n_samples, 4)
    indep2 = np.random.randn(n_samples, 5)

    # Create two datasets, with each dimension composed as
    # a sum of 75% one of the latent variables and 25% independent component
    X = .25*indep1 + .75*np.vstack((L1, L2, L1, L2)).T
    Y = .25*indep2 + .75*np.vstack((L1, L2, L1, L2, L1)).T

    # Split each dataset into two halves: training set and test set
    train = range(0, len(X), 2)
    test = range(1, len(X), 2)

    kcca = KernelCCA(reg=1, kernel='linear')
    print(kcca.fit(X[train], Y[train]).score(X[test], Y[test]))

    kcca = KernelCCA(reg=1, kernel='rbf')
    kcca.fit(X[train], Y[train]).score(X[test], Y[test])
