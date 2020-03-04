import mne
import numpy as np
import os.path as op
from sklearn.preprocessing import scale
from common import (add_part_of_speech, add_word_frequency, add_word_length,
                    read_log, get_log_times, data_path)


def fetch_data(log_file):
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
    epochs = epochs['condition=="word"']

    epochs = epochs.apply_baseline(None, 0).crop(-.2, 1.)

    # Preprocessing features
    df = add_word_frequency(epochs.metadata[['word', ]])
    df = add_word_length(df)
    df = add_part_of_speech(df)
    df['word_func'] = False
    df.loc[df.query('(ADP+CONJ+DET+PRON)==True').index,
           'word_func'] = True
    word = df.word.apply(lambda s: s.lower())
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        df[letter] = word.str.count(letter) > 1

    df['word_dummy'] = (scale(df['word_length']) +
                        scale(df['word_freq']) +
                        np.random.randn(len(df)))

    meg = epochs.get_data()
    n_samples, n_chans, n_times = meg.shape
    for t in range(n_times):
        meg[:, :, t] = scale(meg[:, :, t])

    names = ['word_length', 'word_freq', 'word_func', 'word_dummy']

    features = scale(df[names].values)

    return meg, epochs.times, features, names, df.word
