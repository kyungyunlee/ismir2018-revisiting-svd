import os
import sys
import numpy as np
import librosa
from config import *
import scipy.io
from scipy.signal import medfilt


def load_label(audio_feat, audio_label_file):
    '''
    Args : 
        audio_feat : computed audio feature. Shape = (116, total_frames)
        audio_label_file : path to the label file 
    Return : 
        label : list of ground truth label. Shape = (total_frames, ) 
    '''
    with open(audio_label_file, 'r') as f:
        total_frames = audio_feat.shape[1]
        print("total frames", total_frames)
        label = np.zeros((total_frames,))

        for line in f:
            l = line.strip('\n').split(' ')

            start = librosa.time_to_frames(float(l[0]), sr=SR, hop_length=HOP_LENGTH, n_fft=FRAME_LEN)[0]
            end = librosa.time_to_frames(float(l[1]), sr=SR, hop_length=HOP_LENGTH, n_fft=FRAME_LEN)[0]
            is_vocal = 1 if l[2] == 'sing' or l[2] == '1' else 0
            if start == -1:
                start = 0
            label[start:end] = is_vocal
    return label


def load_single_xy(audio_file, feat_dir, label_dir):
    ''' Create (audio, label) data pair for the given audio file name
    Args : 
        audio_file : name of the audio feature npy file 
        feat_dir : path to the audio feature npy file 
        label_dir : path to the label files 
    Return :
        audio_feat : computed audio feature of the song. Shape = (116, total_frames)
        label : annotations for the song. Shape = (total_frames, )
    '''
    print(audio_file, feat_dir)
    # load audio    
    '''
    if feat_dir == FEAT_JAMENDO_DIR:
        feat_dir = os.path.join(mel_dir, 'test')
    '''
    audio_feat = np.load(os.path.join(feat_dir, audio_file))  # (frame len, 116)
    audio_feat = audio_feat.swapaxes(0, 1)
    print(audio_feat.shape)

    # load label 
    if label_dir == '':
        label = np.zeros((audio_feat.shape[1]))
    else:
        audio_file_name = audio_file.split('.')[0]
        audio_label_file = os.path.join(label_dir, audio_file_name + '.lab')
        label = load_label(audio_feat, audio_label_file)
        assert audio_feat.shape[1] == len(label)

    return audio_feat, label


def load_xy_data(song, feat_dir, label_dir, train_valid=None):
    '''
    Args : 
        song : name of the song. None if train_valid =='train' or 'valid'
        feat_dir : path to the dir of saved audio features 
        label_dir : path to the label files 
        train_valid : 'train' or 'valid' or none 
    Return :
        total_x : 
        total_y : 
    '''
    train_valid_set = ['train', 'valid']
    total_x = []
    total_y = []

    if train_valid in train_valid_set:
        for audio_file in os.listdir(os.path.join(feat_dir, train_valid)):
            x, y = load_single_xy(audio_file, os.path.join(feat_dir, train_valid), label_dir)
            num_input = x.shape[1]
            for i in range(0, num_input - INPUT_FRAME_LEN, INPUT_HOP):
                x_segment = x[:, i:i + INPUT_FRAME_LEN]
                # pick the center frame label 
                y_label = y[i + INPUT_FRAME_LEN // 2]
                x_segment = x_segment.reshape(-1)  # flatten

                total_x.append(x_segment)
                total_y.append(y_label)
        total_x = np.array(total_x)
        total_y = np.array(total_y)
        assert len(total_x) == len(total_y)
        return total_x, total_y

    else:
        audio_file = song
        print(audio_file)
        x, y = load_single_xy(audio_file, feat_dir, label_dir)
        num_input = x.shape[1]
        for i in range(0, num_input - INPUT_FRAME_LEN, INPUT_HOP):
            x_segment = x[:, i:i + INPUT_FRAME_LEN]
            # pick the center frame label 
            y_label = y[i + INPUT_FRAME_LEN // 2]
            x_segment = x_segment.reshape(-1)  # flatten

            total_x.append(x_segment)
            total_y.append(y_label)
        total_x = np.array(total_x)
        total_y = np.array(total_y)
        return total_x, total_y


# ------- MedleyDB ------- #

def load_label_mdb(audio_feature, label_file):
    frame_level_label = np.zeros((audio_feature.shape[1],))
    sample_level_label = np.load(label_file)
    for i in range(len(frame_level_label)):
        t1 = librosa.frames_to_time(i, sr=SR, hop_length=HOP_LENGTH)
        f2 = librosa.time_to_samples(t1, sr=22050)
        if f2 >= len(sample_level_label):
            continue
        frame_level_label[i] = sample_level_label[f2]

    return frame_level_label


def load_single_xy_mdb(audio_file, feat_dir, label_dir):
    '''
    Args : 
        audio_file : name of the audio feature npy file
        feat_dir : path to the folder containing audio feature npy files
        label_dir : path to the label dir
    Return :
        audio_feat : loaded audio feature of the song. Shape=(116, total_frames)
        label : annotation of the song. Shape=(total_frames, )
    '''
    audio_feat = np.load(os.path.join(feat_dir, audio_file))  # (frame len, 116)
    audio_feat = audio_feat.swapaxes(0, 1)
    print(audio_feat.shape)

    audio_label_file = audio_file.split('.')[0].replace("MIX", "RAW")
    audio_label_file = audio_label_file.replace("p0", "RAW")
    audio_label_file = audio_label_file.replace("p6", "RAW")
    audio_label_file = audio_label_file.replace("p12", "RAW")
    audio_label_file = audio_label_file.replace("m6", "RAW")
    audio_label_file = audio_label_file.replace("m12", "RAW")
    audio_label_file = os.path.join(label_dir, audio_label_file)
    label = load_label_mdb(audio_feat, audio_label_file + '.npy')
    return audio_feat, label


def load_xy_data_mdb(song, feat_dir, label_dir):
    '''
    Args :
        song : name of the song 
        feat_dir : path to the audio feature dir
        label_dir : path to the label dir 
    Return :
        total_x :
        total_y : 
    '''
    total_x = []
    total_y = []
    x, y = load_single_xy_mdb(song, feat_dir, label_dir)
    num_input = x.shape[1]
    print("len, num input", x.shape[1], num_input)
    for i in range(0, num_input - INPUT_FRAME_LEN, INPUT_HOP):
        x_segment = x[:, i: i + INPUT_FRAME_LEN]
        # pick the center frame label 
        y_label = y[i + INPUT_FRAME_LEN // 2]
        # flatten
        x_segment = x_segment.reshape(-1)

        total_x.append(x_segment)
        total_y.append(y_label)
    total_x = np.array(total_x)
    total_y = np.array(total_y)
    return total_x, total_y
