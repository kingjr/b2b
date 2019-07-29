import os.path as op
import glob
import numpy as np
import pandas as pd
import subprocess
import spacy
from wordfreq import zipf_frequency as word_frequency

# Setup paths and file names
this_path = '/' + op.join(*op.realpath(__file__).split('/')[:-1])


def setup_datapath():
    fname_datapath = '/' + this_path + '/data_path.txt'

    if not op.isfile(fname_datapath):
        data_path = input("Enter data path (or create a data_path.txt):")
        if data_path[-1] != '/':
            data_path += '/'
        with open(fname_datapath, 'w') as f:
            f.write(data_path)
    with open(fname_datapath, 'r') as f:
        data_path = f.read()
    return data_path


def setup_logfiles():
    fname_logfiles = '/' + this_path + '/log_files.csv'
    data_path = setup_datapath()
    if not op.isfile(fname_logfiles):
        tasks = dict(visual='Vis', auditory='Aud')
        log_files = list()

        for task in ('visual', 'auditory'):

            log_path = data_path + 'sourcedata/meg_task/'
            files = glob.glob(log_path + "*-MEG-MOUS-%s*.log" % tasks[task])

            for file in np.sort(files):
                subject = file.split('-')[0].split('/')[-1]
                log_files.append(dict(
                    subject=int(subject[1:]),
                    task=task,
                    log_id=int(file.split('-')[1]),
                    log_file=op.join('sourcedata', 'meg_task',
                                     file.split('/')[-1]),
                    meg_file=op.join('sub-' + subject, 'meg',
                                     'sub-%s_task-%s_meg.ds' % (subject, task))
                    )
                )
        log_files = pd.DataFrame(log_files)
        # Remove corrupted log
        log_files = log_files.loc[(log_files.subject != 1006) &
                                  (log_files.subject != 1017)]
        log_files.to_csv(fname_logfiles)
    return pd.read_csv(fname_logfiles)


def setup_stimuli():
    fname_stimuli = '/' + this_path + '/stimuli.csv'
    data_path = setup_datapath()
    if not op.isfile(fname_stimuli):

        source = op.join(data_path, 'stimuli', 'stimuli.txt')

        with open(source, 'r') as f:
            stimuli = f.read()
        while '  ' in stimuli:
            stimuli.replace('  ', ' ')

        # clean up
        stimuli = stimuli.split('\n')[:-1]
        stim_id = [int(s.split(' ')[0]) for s in stimuli]
        sequences = [' '.join(s.split(' ')[1:]) for s in stimuli]
        stimuli = pd.DataFrame([dict(index=idx, sequence=seq)
                                for idx, seq in zip(stim_id, sequences)])
        stimuli.to_csv(fname_stimuli, index=False)
    return pd.read_csv(fname_stimuli, index_col='index')


def setup_morphemes():
    # download
    commands = ('polyglot download morph2.en morph2.ar',
                'polyglot download morph2.nl')
    for command in commands:
        subprocess.Popen(command.split(), stdout=subprocess.PIPE)


data_path = setup_datapath()
log_files = setup_logfiles()
stimuli = setup_stimuli()


def _parse_log(log_fname):
    with open(log_fname, 'r') as f:
        text = f.read()

    # Fixes broken inputs
    text = text.replace('.\n', '.')

    # file is made of two blocks
    block1, block2 = text.split('\n\n\n')

    # read first header
    header1 = block1.replace(' ', '_').split('\n')[3].split('\t')
    header1[6] = 'time_uncertainty'
    header1[8] = 'duration_uncertainty'

    # read first data
    df1 = pd.DataFrame([s.split('\t') for s in block1.split('\n')][5:],
                       columns=header1)
    # the two dataframe are only synced on certains rows
    common_samples = ('Picture', 'Sound', 'Nothing')
    sel = df1['Event_Type'].apply(lambda x: x in common_samples)
    index = df1.loc[sel].index

    # read second header
    header2 = block2.replace(' ', '_').split('\n')[0].split('\t')
    header2[7] = 'time_uncertainty'
    header2[9] = 'duration_uncertainty'

    # read second data
    df2 = pd.DataFrame([s.split('\t') for s in block2.split('\n')[2:-1]],
                       columns=header2, index=index)

    # remove duplicate
    duplicates = np.intersect1d(df1.keys(), df2.keys())
    for key in duplicates:
        assert (df1.loc[index, key] == df2[key].fillna('')).all()
        df2.pop(key)

    log = pd.concat((df1, df2), axis=1)
    return log


def _clean_log(log):
    # Relabel condition: only applies to sample where condition changes
    translate = dict(
        ZINNEN='sentence',
        WOORDEN='word_list',
        FIX='fix',
        QUESTION='question',
        Response='response',
        ISI='isi',
        blank='blank',
    )
    for key, value in translate.items():
        sel = log.Code.astype(str).str.contains(key)
        log.loc[sel, 'condition'] = value
    log.loc[log.Code == '', 'condition'] = 'blank'

    # Annotate sequence idx and extend context to all trials
    start = 0
    block = 0
    context = 'init'
    log['new_context'] = False
    query = 'condition in ("word_list", "sentence")'
    for idx, row in log.query(query).iterrows():
        log.loc[start:idx, 'context'] = context
        log.loc[start:idx, 'block'] = block
        log.loc[idx, 'new_context'] = True
        context = row.condition
        block += 1
        start = idx
    log.loc[start:, 'context'] = context
    log.loc[start:, 'block'] = block

    # Format time
    log['time'] = 0
    idx = log.Time.str.isnumeric() == True  # noqa
    log.loc[idx, 'time'] = log.loc[idx, 'Time'].astype(float) / 1e4

    # Extract individual word
    log.loc[log.condition.isna(), 'condition'] = 'word'
    idx = log.condition == 'word'
    words = log.Code.str.strip('0123456789 ')
    log.loc[idx, 'word'] = words.loc[idx]
    sel = log.query('word=="" and condition=="word"').index
    log.loc[sel, 'word'] = pd.np.nan
    log.loc[log.word.isna() & (log.condition == "word"), 'condition'] = 'blank'
    return log


def _add_stim_id(log, verbose):
    # Find beginning of each sequence (word list or sentence)
    start = 0
    sequence_pos = -1
    for idx, row in log.query('condition == "fix"').iterrows():
        if sequence_pos >= 0:
            log.loc[start:idx, 'sequence_pos'] = sequence_pos
        sequence_pos += 1
        start = idx
    log.loc[start:, 'sequence_pos'] = sequence_pos

    # Find corresponding stimulus id
    stim_id = 0
    lower30 = lambda s: s[:30].lower()  # noqa
    stimuli['first_30_chars'] = stimuli.sequence.apply(lower30)
    sel = slice(0, 0)
    for pos, row in log.groupby('sequence_pos'):
        if pos == -1:
            continue

        # select words in this sequence
        sel = row.condition == "word"
        if not sum(sel):
            continue

        # match with stimuli
        first_30_chars = ' '.join(row.loc[sel, 'word'])[:30].lower()  # noqa
        stim_id = stimuli.query('first_30_chars == @first_30_chars').index
        assert len(stim_id) == 1
        stim_id = stim_id[0]

        n_words = len(stimuli.loc[stim_id, 'sequence'].split(' '))
        if (n_words != sum(sel)) and verbose:
            print('mistach of %i words in %s (stim %i)' % (n_words - sum(sel),
                                                           pos, stim_id))
            print('stim: %s' % stimuli.loc[stim_id, 'sequence'])
            print('log: %s' % ' '.join(row.loc[sel, 'word']))

        # Update
        log.loc[row.index, 'stim_id'] = stim_id
    return log


def read_log(log_fname, task='auto', verbose=False):
    log = _parse_log(log_fname)
    log = _clean_log(log)
    if task == 'auto':
        task = 'visual' if log_fname[-7:] == 'Vis.log' else 'auditory'
    if task == 'visual':
        # TODO: add sequence annotation for auditory
        log = _add_stim_id(log, verbose=verbose)
    return log


def get_log_times(log, events, sfreq):
    sel = np.sort(np.r_[
        np.where(events[:, 2] == 20)[0],  # fixation
        np.where(events[:, 2] == 10)[0]  # context
    ])
    common_megs = events[sel]
    common_logs = log.query('(new_context == True) or condition=="fix"')

    last_log = common_logs.time[0]
    last_meg = common_megs[0, 0]
    last_idx = 0
    assert len(common_megs) == len(common_logs)
    for common_meg, (idx, common_log) in zip(
            common_megs, common_logs.iterrows()):

        if common_meg[2] == 20:
            assert common_log.condition == 'fix'
        else:
            assert common_log.condition in ('sentence', 'word_list')

        log.loc[idx, 'meg_time'] = common_meg[0] / sfreq

        sel = slice(last_idx+1, idx)
        times = log.loc[sel, 'time'] - last_log + last_meg / sfreq
        assert np.all(np.isfinite(times))
        log.loc[sel, 'meg_time'] = times

        last_log = common_log.time
        last_meg = common_meg[0]
        last_idx = idx

        assert np.isfinite(last_log) * np.isfinite(last_meg)

    # last block
    sel = slice(last_idx+1, None)
    times = log.loc[sel, 'time'] - last_log + last_meg / sfreq
    log.loc[sel, 'meg_time'] = times
    log['meg_sample'] = np.array(log.meg_time.values * sfreq, int)
    return log


def match_list(A, B, on_replace='delete'):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """
    from Levenshtein import editops

    unique = np.unique(np.r_[A, B])
    label_encoder = dict((k, v) for v, k in enumerate(unique))

    def int_to_unicode(array):
        return ''.join([str(chr(label_encoder[ii])) for ii in array])

    changes = editops(int_to_unicode(A), int_to_unicode(B))
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type, val_a, val_b in changes:
        if type == 'insert':
            B_sel[val_b] = np.nan
        elif type == 'delete':
            A_sel[val_a] = np.nan
        elif on_replace == 'delete':
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == 'keep':
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


def add_part_of_speech(df):
    sentences = ' '.join(df['word'].values)
    while '  ' in sentences:
        sentences = sentences.replace('  ', ' ')

    nlp = spacy.load("nl_core_news_sm")
    doc = [i for i in nlp(sentences)]
    from_idx, to_idx = match_list([i.text for i in doc],
                                  df.word.values)

    part_of_speech = [doc[i].pos_ for i in from_idx]
    idx = df.index.values[to_idx]
    df.loc[idx, 'pos'] = part_of_speech
    df.loc[df.pos == 'X', 'pos'] = pd.np.nan
    #categories = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
                  # 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB']
    #df = df.join(pd.get_dummies(df.pos, columns=categories))
    df = df.join(pd.get_dummies(df.pos))
    if 'PROPN' not in df.keys():
        verb_idx = df.columns.get_loc("VERB")
        df.insert(verb_idx, 'PROPN', 0)

    return df


def add_word_frequency(df):
    freq = df.word.apply(lambda word: word_frequency(word, 'nl'))  # noqa
    df['word_freq'] = freq
    return df


def add_word_length(df):
    df['word_length'] = df.word.astype(str).apply(len)
    return df


def add_letter_count(df):
    word = df.word.str
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        df[letter] = word.count(letter)
    return df


def read_mri_events(event_fname):
    """This is needs to be enriched depending on the analysis
    """
    # Read MRI events
    events = pd.read_csv(event_fname, sep='\t')

    # Add context: sentence or word list?
    contexts = dict(WOORDEN='word_list', ZINNEN='sentence')
    for key, value in contexts.items():
        sel = events.value.str.contains(key)
        events.loc[sel, 'context'] = value
        events.loc[sel, 'condition'] = value

    # Clean up MRI event mess
    sel = ~events.context.isna()
    start = 0
    context = 'init'
    for idx, row in events.loc[sel].iterrows():
        events.loc[start:idx, 'context'] = context
        start = idx
        context = row.context
    events.loc[start:, 'context'] = context

    # Add event condition: word, blank, inter stimulus interval etc
    conditions = (('50', 'pulse'), ('blank', 'blank'), ('ISI', 'isi'))
    for key, value in conditions:
        sel = events.value == key
        events.loc[sel, 'condition'] = value

    events.loc[events.value.str.contains('FIX '), 'condition'] = 'fix'

    # Extract words from file
    sel = events.condition.isna()
    words = events.loc[sel, 'value'].apply(lambda s: s.strip('0123456789 '))
    events.loc[sel, 'word'] = words

    # Remove empty words
    sel = (events.word.astype(str).apply(len) == 0) & (events.condition.isna())
    events.loc[sel, 'word'] = pd.np.nan
    events.loc[sel, 'condition'] = 'blank'
    events.loc[~events.word.isna(), 'condition'] = 'word'

    # --- Add word frequency
    sel = events.condition == 'word'

    def get_word_freq(word):
        return word_frequency(word, 'en', wordlist='best', minimum=0.0)

    events.loc[sel, 'word_freq'] = events.loc[sel, 'word'].apply(get_word_freq)

    # --- Add word length
    sel = events.condition == 'word'
    events['word_length'] = events.word.astype(str).apply(len)

    # --- TODO Add whatever features may be relevant here

    return events
