import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from models import RegCCA, B2B, score_knockout


def fetch_data():
    # Preprocess data
    data_path = sample.data_path() + '/MEG/sample/'
    raw_fname = data_path + 'sample_audvis_filt-0-40_raw.fif'
    events_fname = data_path + 'sample_audvis_filt-0-40_raw-eve.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    raw.filter(1., 30., fir_design='firwin')
    events_samples, _, events_values = mne.read_events(events_fname).T

    Y = raw.get_data()[picks]
    X = np.zeros((4, raw.n_times))
    for idx, value in enumerate(set(events_values)):
        sel = np.where(events_values == value)[0]
        X[idx, events_samples[sel]] = 1.

    # Robust scaling
    low, high = np.percentile(Y, [.01, 99.99], axis=1)
    for ch, (l, h) in enumerate(zip(low, high)):
        Y[ch] = np.clip(Y[ch], l, h)

    Y = scale(Y.T)
    X = scale(X.T)

    return X, Y


def time_lag(X, tois):

    n_times, n_events = X.shape
    Xs = np.zeros((n_times, n_events, len(tois)))

    for i, t in enumerate(tois):
        Xs[:, :, i] = np.roll(X, t, axis=0)

    return Xs.reshape(n_times, -1)


if __name__ == '__main__':

    X, Y = fetch_data()

    models = dict(B2B=B2B(ensemble=20), CCA=RegCCA())
    tois = range(-10, 10)

    axes = iter(plt.subplots(len(models), 1)[1])

    for m, model in models.items():
        Xs = time_lag(X, tois)

        cv = KFold(5)
        scores = np.zeros((cv.n_splits, Xs.shape[1]))
        for split, (train, test) in enumerate(cv.split(X)):
            print('.', end='')
            model.fit(Xs[train], Y[train])
            scores[split] = score_knockout(model, Xs[test], Y[test])

        scores = scores.mean(0).reshape(-1, len(tois))

        next(axes).plot(scores.T)
