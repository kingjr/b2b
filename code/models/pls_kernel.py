"""A module which implements kernel PLS.

from github.com/jhumphry/regressions/blob/master/examples/kpls_example.py

ISC License

Regressions
===========

Provides various forms of regression which are not all covered by other
Python statistical packages. The aim is to achieve clarity of
implementation with speed a secondary goal. Python 3.5 and Numpy 1.10 or
greater are required as the new '@' matrix multiplication operator is
used.

All of the regressions require the X and Y data to be provided in the form
of matrices, with one row per data sample and the same number of data
samples in each. Currently missing data or NaN are not supported.

"""

# Copyright (c) 2015, James Humphry - see LICENSE file for details

import random
import abc
import numpy as np
import math
import scipy.linalg as linalg


class ParameterError(Exception):
    """Parameters passed to a regression routine are unacceptable

    This is a generic exception used to indicate that the parameters
    passed are mis-matched, nonsensical or otherwise problematic.
    """
    pass


class ConvergenceError(Exception):
    """Iterative algorithm failed to converge.

    Many of the routines used for regressions are iterative and in some
    cases may not converge. This is mainly likely to happen if the data
    has pathological features, or if too many components of a data set
    have been extracted by an iterative process and the residue is
    becoming dominated by rounding or other errors.
    """
    pass


DEFAULT_MAX_ITERATIONS = 250
"""Default maximum number of iterations that iterative routines will
attempt before raising a ConvergenceError."""

DEFAULT_EPSILON = 1.0E-6
"""A default epsilon value used in various places, such as to decide when
iterations have converged sufficiently."""


class RegressionBase(metaclass=abc.ABCMeta):

    """Abstract base class for regressions

    All the various types of regression objects will have at least the
    attributes present here.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        standardize_X (boolean, optional): Standardize the X data
        standardize_Y (boolean, optional): Standardize the Y data

    Attributes:
        data_samples (int): number of calibration data samples (=N)
        max_rank (int): maximum rank of calibration X-data (limits the
            number of components that can be found)
        X_variables (int): number of X variables (=n)
        Y_variables (int): number of Y variables (=m)
        X_offset (float): Offset of calibration X data from zero
        Y_offset (float): Offset of calibration Y data from zero
        standardized_X (boolean): whether X data had variance standardized
        standardized_Y (boolean): whether Y data had variance standardized
        X_rscaling (float): the reciprocal of the scaling factor used for X
        Y_scaling (float): the scaling factor used for Y
    """

    @abc.abstractmethod
    def __init__(self, X, Y, standardize_X=False, standardize_Y=False):
        pass

    def _prepare_data(self, X, Y, standardize_X=False, standardize_Y=False):

        """A private method that conducts routine data preparation

        Sets all of the RegressionBase attributes on ``self`` and returns
        suitably centred and (where requested) variance standardized X and
        Y data.

        Args:
            X (ndarray N x n): X calibration data, one row per data sample
            Y (ndarray N x m): Y calibration data, one row per data sample
            standardize_X (boolean, optional): Standardize the X data
            standardize_Y (boolean, optional): Standardize the Y data

        Returns:
            Xc (ndarray N x n): Centralized and standardized X data
            Yc (ndarray N x m): Centralized and standardized Y data

        """

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same number of '
                                 'rows (data samples)')

        # Change 1-D arrays into column vectors
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))

        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]
        self.standardized_X = standardize_X
        self.standardized_Y = standardize_Y

        self.X_offset = X.mean(0)
        Xc = X - self.X_offset

        if standardize_X:
            # The reciprocals of the standard deviations of each column are
            # stored as these are what are needed for fast prediction
            self.X_rscaling = 1.0 / Xc.std(0, ddof=1)
            Xc *= self.X_rscaling
        else:
            self.X_rscaling = None

        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset
        if standardize_Y:
            self.Y_scaling = Y.std(0, ddof=1)
            Yc /= self.Y_scaling
        else:
            self.Y_scaling = None

        return Xc, Yc


class Kernel_PLS(RegressionBase):

    """Non-linear Kernel PLS regression using the PLS2 algorithm

    This class implements kernel PLS regression by transforming the input
    X data into feature space by applying a kernel function between each
    pair of inputs. The kernel function provided will be called with two
    vectors and should return a float. Kernels should be symmetrical with
    regard to the order in which the vectors are supplied. The PLS2
    algorithm is then applied to the transformed data. The application of
    the kernel function means that non-linear transformations are
    possible.

    Note:
        If ``ignore_failures`` is ``True`` then the resulting object
        may have fewer components than requested if convergence does
        not succeed.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        X_kernel (function): Kernel function
        max_iterations (int, optional) : Maximum number of iterations of
            NIPALS to attempt
        iteration_convergence (float, optional): Difference in norm
            between two iterations at which point the iteration will be
            considered to have converged.
        ignore_failures (boolean, optional): Do not raise an error if
            iteration has to be abandoned before the requested number
            of components have been recovered

    Attributes:
        components (int): number of components extracted (=g)
        X_training_set (ndarray N x n): X calibration data (centred)
        K (ndarray N x N): X calibration data transformed into feature space
        P (ndarray n x g): Loadings on K (Components extracted from data)
        Q (ndarray m x g): Loadings on Y (Components extracted from data)
        T (ndarray N x g): Scores on K
        U (ndarray N x g): Scores on Y
        B_RHS (ndarray n x m): Partial regression matrix

    """

    def __init__(self, X, Y, g, X_kernel,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        Xc, Yc = super()._prepare_data(X, Y)

        self.X_training_set = Xc
        self.X_kernel = X_kernel

        K = np.empty((self.data_samples, self.data_samples))
        for i in range(0, self.data_samples):
            for j in range(0, i):
                K[i, j] = X_kernel(Xc[i, :], Xc[j, :])
                K[j, i] = K[i, j]
            K[i, i] = X_kernel(Xc[i, :], Xc[i, :])

        centralizer = (np.identity(self.data_samples)) - \
            (1.0 / self.data_samples) * \
            np.ones((self.data_samples, self.data_samples))
        K = centralizer @ K @ centralizer
        self.K = K

        T = np.empty((self.data_samples, g))
        Q = np.empty((self.Y_variables, g))
        U = np.empty((self.data_samples, g))
        P = np.empty((self.data_samples, g))

        self.components = 0
        K_j = K
        Y_j = Yc

        for j in range(0, g):
            u_j = Y_j[:, random.randint(0, self.Y_variables-1)]

            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and \
                    iteration_change > iteration_convergence:

                w_j = K_j @ u_j
                t_j = w_j / np.linalg.norm(w_j, 2)

                q_j = Y_j.T @ t_j

                old_u_j = u_j
                u_j = Y_j @ q_j
                u_j /= np.linalg.norm(u_j, 2)
                iteration_change = linalg.norm(u_j - old_u_j)
                iteration_count += 1

            if iteration_count >= max_iterations:
                if ignore_failures:
                    break
                else:
                    raise ConvergenceError('PLS2 failed to converge for '
                                           'component: '
                                           '{}'.format(self.components+1))

            T[:, j] = t_j
            Q[:, j] = q_j
            U[:, j] = u_j

            P[:, j] = (K_j.T @ w_j) / (w_j @ w_j)
            deflator = (np.identity(self.data_samples) - np.outer(t_j.T, t_j))
            K_j = deflator @ K_j @ deflator
            Y_j = Y_j - np.outer(t_j, q_j.T)
            self.components += 1

        # If iteration stopped early because of failed convergence, only
        # the actual components will be copied

        self.T = T[:, 0:self.components]
        self.Q = Q[:, 0:self.components]
        self.U = U[:, 0:self.components]
        self.P = P[:, 0:self.components]

        self.B_RHS = self.U @ linalg.inv(self.T.T @ self.K @ self.U) @ self.Q.T

    def prediction(self, Z):
        """Predict the output resulting from a given input

        Args:
            Z (ndarray of floats): The input on which to make the
                prediction. A one-dimensional array will be interpreted as
                a single multi-dimensional input unless the number of X
                variables in the calibration data was 1, in which case it
                will be interpreted as a set of inputs. A two-dimensional
                array will be interpreted as one multi-dimensional input
                per row.

        Returns:
            Y (ndarray of floats) : The predicted output - either a one
            dimensional array of the same length as the number of
            calibration Y variables or a two dimensional array with the
            same number of columns as the calibration Y data and one row
            for each input row.
        """

        if len(Z.shape) == 1:
            if self.X_variables == 1:
                Z = Z.reshape((Z.shape[0], 1))
                Kt = np.empty((Z.shape[0], self.data_samples))
            else:
                if Z.shape[0] != self.X_variables:
                    raise ParameterError('Data provided does not have the '
                                         'same number of variables as the '
                                         'original X data')
                Z = Z.reshape((1, Z.shape[0]))
                Kt = np.empty((1, self.data_samples))
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            Kt = np.empty((Z.shape[0], self.data_samples))

        for i in range(0, Z.shape[0]):
            for j in range(0, self.data_samples):
                Kt[i, j] = self.X_kernel(Z[i, :] - self.X_offset,
                                         self.X_training_set[j, :])

        centralizer = (1.0 / self.data_samples) * \
            np.ones((Z.shape[0], self.data_samples))

        Kt = (Kt - centralizer @ self.K) @ \
             (np.identity(self.data_samples) -
              (1.0 / self.data_samples) * np.ones(self.data_samples))

        # Fix centralisation - appears to be necessary but not usually
        # mentioned in papers

        Kt -= Kt.mean(0)

        return self.Y_offset + Kt @ self.B_RHS


def std_gaussian(x, y):
    """A Gaussian kernel with width 1.
    The Gaussian kernel with standard deviation 1 is a routine choice.
    Args:
        x (float or numpy.ndarray of float): The x coordinate
        y (float or numpy.ndarray of float): The y coordinate
    """

    return 0.3989422804014327 * math.exp(- 0.5 * np.sum((x-y)**2))


def make_gaussian_kernel(width=1.0):
    """Create a Gaussian kernel with adjustable width
    Args:
        width (float) : The standard deviation of the Gaussian function
            which adjusts the width of the resulting kernel.
    Returns:
        gaussian_kernel (function) : A function of two floats or
        numpy.ndarray of floats which computes the Gaussian kernel of
        the desired width.
    """

    normalization = 1.0 / math.sqrt(2.0 * math.pi * width)
    scale = 1.0 / (2.0 * width**2)

    def gaussian_kernel(x, y):
        return normalization * math.exp(-scale * np.sum((x-y)**2))

    return gaussian_kernel


class KernelPLS(object):
    """sklearn API"""
    def __init__(self, n_components=2, kernel='rbf'):
        self.kernel = kernel
        self.n_components = n_components

    def fit(self, X, Y):
        if self.kernel == 'rbf':
            kernel = make_gaussian_kernel()
        else:
            kernel = self.kernel
        self.estimator_ = Kernel_PLS(X, Y, self.n_components, X_kernel=kernel)
        return self

    def predict(self, X):
        return self.estimator_.prediction(X)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


if __name__ == '__main__':
    from sklearn.cross_decomposition import PLSRegression
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

    kpls = KernelPLS(n_components=2, kernel='rbf')
    print(kpls.fit(X[train], Y[train]).score(X[test], Y[test]))

    pls = PLSRegression(n_components=2)
    print(pls.fit(X[train], Y[train]).score(X[test], Y[test]))
