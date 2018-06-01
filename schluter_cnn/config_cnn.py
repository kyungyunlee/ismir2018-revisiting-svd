'''
Config file for CNN model 
'''

# --  CHANGE PATHS --# 

# Jamendo
JAMENDO_DIR = '../jamendo/'  # path to jamendo dataset
MEL_JAMENDO_DIR = '../jamendo/schluter_mel_dir/'  # path to save computed melgrams of jamendo
JAMENDO_LABEL_DIR = '../jamendo/jamendo_lab/'  # path to jamendo dataset label

# MedleyDB
MDB_VOC_DIR = '/media/bach1/dataset/MedleyDB/'  # path to medleyDB vocal containing songs
MDB_LABEL_DIR = '../mdb_voc_label/'

# vibrato 
SAW_DIR = '../sawtooth_200/songs/'
MEL_SAW_DIR = '../sawtooth_200/schluter_mel_dir/'

# -- Audio processing parameters --#

SR = 22050
FRAME_LEN = 1024
HOP_LENGTH = 315
CNN_INPUT_SIZE = 115  # 1.6 sec
CNN_OVERLAP = 5  # Hopsize of 5 for training, 1 for inference
N_MELS = 80
CUTOFF = 8000  # fmax = 8kHz

# -- CNN model parameters --#

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.2
NUM_EPOCHS = 50

EARLY_STOPPING = 5
REDUCE_LR = 3

THRESHOLD = 0.5  # threshold for binary classification
