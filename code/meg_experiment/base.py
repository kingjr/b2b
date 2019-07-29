import mne
import numpy as np
import os.path as op
from scipy.linalg import svd
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
from common import (add_part_of_speech, add_word_frequency, add_word_length,
                    read_log, get_log_times, data_path)


def fetch_data(log_file):
    fname = op.join(data_path, '%s-epo.fif' % log_file.subject)
    if op.isfile(fname):
        epochs = mne.read_epochs(fname, preload=True)
    else:
        # file name
        raw_fname = op.join(data_path, log_file['meg_file'])
        log_fname = op.join(data_path, log_file['log_file'])

        # preprocess meg data
        raw = mne.io.read_raw_ctf(raw_fname, preload=True)
        raw.filter(.1, 40.)
        events = mne.find_events(raw)

        # preprocess annotations
        log = read_log(log_fname)

        # link meg and annotations
        log = get_log_times(log, events,  raw.info['sfreq'])

        # Segment into word-locked epochs
        log_events = np.c_[log.meg_sample, np.ones((len(log), 2), int)]
        _, idx = np.unique(log_events[:, 0], return_index=True)
        picks = mne.pick_types(raw.info, meg=True, eeg=False,
                               stim=False, eog=False, ecg=False)
        epochs = mne.Epochs(
            raw,
            events=log_events[idx], metadata=log.iloc[idx],
            tmin=-.100, tmax=1.5, decim=20,
            picks=picks, preload=True,
        )
        epochs.save(fname)

    epochs = epochs['condition=="word"']

    epochs = epochs.apply_baseline(None, 0).crop(-.2, 1.)

    # Preprocessing features
    df = add_word_frequency(epochs.metadata[['word', ]])
    df = add_word_length(df)
    df = add_part_of_speech(df)
    word = df.word.apply(lambda s: s.lower())
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        df[letter] = word.str.count(letter) > 1

    meg = epochs.get_data()
    n_samples, n_chans, n_times = meg.shape
    for t in range(n_times):
        meg[:, :, t] = scale(meg[:, :, t])

    columns = ['word_freq', 'word_length']
    names = [k for k in df.keys() if k not in ('word', 'pos')]
    features = scale(df[names].values)

    return meg, epochs.times, features, names


def ridge_cv(X, Y, alphas, independent_alphas=False, Uv=None):
    """ Similar to sklearn RidgeCV but
   (1) can optimize a different alpha for each column of Y
   (2) return leave-one-out Y_hat
   """
    if isinstance(alphas, (float, int)):
        alphas = np.array([alphas, ], np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, n_x = X.shape
    n, n_y = Y.shape
    # Decompose X
    if Uv is None:
        U, s, _ = svd(X, full_matrices=0)
        v = s**2
    else:
        U, v = Uv
    UY = U.T @ Y

    # For each alpha, solve leave-one-out error coefs
    cv_duals = np.zeros((len(alphas), n, n_y))
    cv_errors = np.zeros((len(alphas), n, n_y))
    for alpha_idx, alpha in enumerate(alphas):
        # Solve
        w = ((v + alpha) ** -1) - alpha ** -1
        c = U @ np.diag(w) @ UY + alpha ** -1 * Y
        cv_duals[alpha_idx] = c

        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        G_diag = (w * U ** 2).sum(axis=-1) + alpha ** -1
        error = c / G_diag[:, np.newaxis]
        cv_errors[alpha_idx] = error

    # identify best alpha for each column of Y independently
    if independent_alphas:
        best_alphas = (cv_errors ** 2).mean(axis=1).argmin(axis=0)
        duals = np.transpose([cv_duals[b, :, i]
                              for i, b in enumerate(best_alphas)])
        cv_errors = np.transpose([cv_errors[b, :, i]
                                  for i, b in enumerate(best_alphas)])
    else:
        _cv_errors = cv_errors.reshape(len(alphas), -1)
        best_alphas = (_cv_errors ** 2).mean(axis=1).argmin(axis=0)
        duals = cv_duals[best_alphas]
        cv_errors = cv_errors[best_alphas]

    coefs = duals.T @ X
    Y_hat = Y - cv_errors
    return coefs, best_alphas, Y_hat


def fit_knockout(X, Y, alphas, knockout=True, independent_alphas=False,
                 pairwise=False):
    """Fit model using all but knockout X features"""
    n_x, n_y = X.shape[1], Y.shape[1]
    knockout = np.arange(n_x)[:, None] if knockout is True else knockout

    K = list()
    for xi in knockout:
        not_xi = np.setdiff1d(np.arange(n_x), xi)

        # When X and Y are matched in dimension, we can speed things
        # up by only computing the regression pairwise, as the other
        # columns won't be analyzed
        if pairwise:
            y_sel = np.asarray(xi)
        else:
            y_sel = np.arange(Y.shape[1])
        x_sel = np.array(not_xi)

        K_ = np.zeros((n_y, n_x))
        K_[y_sel[:, None], x_sel] = ridge_cv(X[:, x_sel], Y[:, y_sel],
                                             alphas, independent_alphas)[0]
        K.append(K_)
    return K


def r_score(Y_true, Y_pred):
    """column-wise correlation coefficients"""
    if Y_true.ndim == 1:
        Y_true = Y_true[:, None]
    if Y_pred.ndim == 1:
        Y_pred = Y_pred[:, None]
    R = np.zeros(Y_true.shape[1])
    for idx, (y_true, y_pred) in enumerate(zip(Y_true.T, Y_pred.T)):
        R[idx] = pearsonr(y_true, y_pred)[0]
    return R


class B2B():
    def __init__(self, alphas=np.logspace(-4, 4, 20),
                 independent_alphas=True,
                 knockout=False):
        self.alphas = alphas
        self.knockout = knockout
        self.independent_alphas = independent_alphas

    def fit(self, X, Y):
        # Fit decoder
        self.G_, G_alpha, YG = ridge_cv(
            Y, X, self.alphas, self.independent_alphas)
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, YG, self.alphas,
                                       self.independent_alphas)
        # Fit knock-out encoders
        if self.knockout:
            self.Ks_ = fit_knockout(X, YG, self.alphas,
                                    self.knockout,
                                    self.independent_alphas, True)

        return self

    def score(self, X, Y):
        # Transform with decoder
        YG = Y @ self.G_.T
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column X
        return r_score(YG, XH)

    def score_knockout(self, X, Y):
        # Transform with decoder
        YG = Y @ self.G_.T
        # Allow knocking out blocks of features
        if self.knockout is True:
            knockout = range(X.shape[1])
        else:
            knockout = self.knockout

        # For each knock-out, compute R score
        K_scores = list()
        for xi, K in zip(knockout, self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            r = r_score(YG[:, xi], XK[:, xi])
            # If one dimension, make it float
            if isinstance(xi, int):
                r = r[0]
            # Difference between standard and knocked-out scores
            K_scores.append(r)

        return np.array(K_scores)


class Forward():
    def __init__(self, alphas=np.logspace(-4, 4, 20), independent_alphas=True,
                 knockout=True):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.knockout = knockout

    def fit(self, X, Y):
        # Fit encoder
        self.H_, H_alpha, _ = ridge_cv(X, Y, self.alphas,
                                       self.independent_alphas)
        # Fit knock-out encoders
        if self.knockout:
            self.Ks_ = fit_knockout(X, Y, self.alphas,
                                    self.knockout,
                                    self.independent_alphas, False)

        return self

    def score(self, X, Y):
        # Make standard and knocked-out encoders predictions
        XH = X @ self.H_.T
        # Compute R for each column of Y
        return r_score(Y, XH)

    def score_knockout(self, X, Y):
        if self.knockout is True:
            knockout = range(X.shape[1])
        else:
            knockout = self.knockout

        R = list()
        for xi, K in zip(knockout, self.Ks_):
            # Knocked-out encoder predictions
            XK = X @ K.T
            # R score for each relevant dimensions of X
            K_score = r_score(Y, XK)
            # Difference between standard and knocked-out scores
            R.append(K_score)

        return np.array(R)
