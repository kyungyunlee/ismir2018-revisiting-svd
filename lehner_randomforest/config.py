BASE_DIR = '../'
JAMENDO_DIR = '../jamendo/'  # path to jamendo dataset
JAMENDO_LABEL_DIR = '../kylee/jamendo/jamendo_lab/'  # path to jamendo label file
MATLAB_JAMENDO_DIR = '../jamendo/randomforest_matlab_2/'  # path to jamendo features from matlab code
FEAT_JAMENDO_DIR = '../jamendo/randomforest_feat_dir/'

# MedleyDB
MDB_VOC_DIR = '/media/bach1/dataset/MedleyDB/'  # path to medleyDB vocal containg songs
MDB_LABEL_DIR = '../mdb_voc_label/'

# vibrato
SAW_DIR = '../sawtooth_200/songs/'
MATLAB_SAW_DIR = '../sawtooth_200/randomforest_matlab/'
FEAT_SAW_DIR = '../sawtooth_200/randomforest_feat_dir/'

# -- Audio processing parameters --#
SR = 22050
FRAME_LEN = 441  # 100ms
HOP_LENGTH = 441  # 20 ms
MF_WIN_SIZE = 3  # frame length X

INPUT_FRAME_LEN = 10
INPUT_HOP = 1
