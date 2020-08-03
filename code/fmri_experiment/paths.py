import os

# Setup paths and file names
this_path = '/' + os.path.join(*os.path.realpath(__file__).split('/')[:-1])


def setup_datapath():
    fname_datapath = '/' + this_path + '/data_path.txt'

    if not os.path.isfile(fname_datapath):
        data_path = input("Enter data path (or create a data_path.txt):")
        if data_path[-1] != '/':
            data_path += '/'
        with open(fname_datapath, 'w') as f:
            f.write(data_path)
    with open(fname_datapath, 'r') as f:
        data_path = f.read()
    return data_path


data_path = setup_datapath()
preproc_path = data_path[:-1] + '_preproc/'

# Where preprocessed fMRI files will be stored
if not os.path.isdir(preproc_path):
    os.makedirs(preproc_path)
