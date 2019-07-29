from base import fetch_data, B2B, Forward
from common import log_files
from sklearn.model_selection import KFold
import numpy as np
from tqdm import trange


def run(name, meg, features):
    """Loop Granger score for each time sample independently"""
    models = dict(B2B=B2B(knockout=True),
                  Forward=Forward(knockout=True))

    cv = KFold(5, shuffle=True)

    n_features = features.shape[1]
    _, n_channels, n_times = meg.shape

    model = models[name]

    # Depending on the model, we may make predictions on different Y dimensions
    n_y = tuple() if name == 'B2B' else tuple([n_channels, ])

    # Loop across times
    H = np.zeros((cv.n_splits, n_times, n_features, *n_y))
    H_r = np.zeros((cv.n_splits, n_times, n_features, *n_y))
    K_r = np.zeros((cv.n_splits, n_times, n_features, *n_y))

    for t in trange(n_times):
        for split, (train, test) in enumerate(cv.split(features)):
            # Fit model
            model.fit(X=features[train],
                      Y=meg[train, :, t])
            # Retrieve coefficients
            H_ = np.diag(model.H_) if name == 'B2B' else model.H_.T
            H[split, t] = H_
            # Compute standard scores
            H_r[split, t] = model.score(X=features[test],
                                        Y=meg[test, :, t])
            # Compute knock out scores
            K_r[split, t] = model.score_knockout(X=features[test],
                                                 Y=meg[test, :, t])

    # Mean across splits
    H = H.mean(0)
    H_r = H_r.mean(0)
    K_r = K_r.mean(0)
    return H, H_r, K_r


subject = 0
log_file = log_files.iloc[subject]
log_files = log_files.query('task=="visual"')
meg, times, features, names = fetch_data(log_file)
run('B2B', meg, features)
