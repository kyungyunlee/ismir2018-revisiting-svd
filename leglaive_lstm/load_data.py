''' Functions for loading data for training and testing in Keras'''
import os
import sys
import numpy as np
import librosa
from config_rnn import *
from scipy.signal import medfilt


# ========================= JAMENDO & OTHERS ================================ #

def load_label(audio_spec, audio_label_file):
    ''' Process and load label for the given audio
    Args :
        audio_spec : computed hpss+melgram of audio. Shape=(n_bins, total_frames) 
        audio_label_file : path to label file
    Return :
        label : list of ground truth annotations per frame. Shape=(total_frames,)
    '''
    with open(audio_label_file, 'r') as f:
        total_frames = audio_spec.shape[1]
        label = np.zeros((total_frames,))

        for line in f:
            l = line.strip('\n').split(' ')
            start = librosa.time_to_frames(float(l[0]), sr=SR, hop_length=N_HOP2)
            end = librosa.time_to_frames(float(l[1]), sr=SR, hop_length=N_HOP2)
            is_vocal = 1 if l[2] == 'sing' or l[2] == '1' else 0
            label[start:end] = int(is_vocal)

    return label


def load_single_xy(audio_file, mel_dir, label_dir):
    ''' Create (audio feature, label) data pair for the givne audio

    Args : 
        audio_file : name of the audio feature npy file
        mel_dir : path to the folder containing npy files
        label_dir : path to the label folder 

    Return : 
        audio_feature : audio feature. Shape=(80, total_frames)
        label : annotation for the audio. Shape=(total_frames,)
    '''

    # load audio    
    if mel_dir == MEL_JAMENDO_DIR:
        mel_dir = os.path.join(mel_dir, 'test')
    audio_feature = np.load(os.path.join(mel_dir, audio_file))

    # load label    
    if label_dir == '':
        label = np.zeros((audio_feature.shape[1]))
    else:
        audio_file_name = audio_file.split('.')[0]
        audio_label_file = os.path.join(label_dir, audio_file_name + '.lab')
        label = load_label(audio_feature, audio_label_file)

    print(audio_file, audio_feature.shape)
    return audio_feature, label


def load_xy_data(song, mel_dir, label_dir, model_name, train_valid=None):
    ''' Load all x,y pair in the given dataset into a list of x_data and list of y_label which are segmented into RNN_INPUT_SIZE shape.

    Args: 
        song : name of the test song npy file. Set to None if train_valid == 'train' or 'valid'
        mel_dir: path to folder of the saved audio feature npy files
        label_dir : path to the folder of label files
        model_name: name of the current rnn model
        train_valid : 'train' or 'valid' or none 

    Return:
        total_x_norm: Shape=(1,218,80)
        total_y: (1, 218, 1)
    '''
    train_valid_set = ['train', 'valid']

    total_x = []
    total_y = []

    if train_valid in train_valid_set:
        for audio_file in os.listdir(os.path.join(mel_dir, train_valid)):
            x, y = load_single_xy(audio_file, os.path.join(mel_dir, train_valid), label_dir)
            for i in range(0, x.shape[1] - RNN_INPUT_SIZE, RNN_OVERLAP):
                x_segment = x[:, i: i + RNN_INPUT_SIZE]
                y_segment = y[i: i + RNN_INPUT_SIZE]
                total_x.append(x_segment)
                total_y.append(y_segment)

        total_x = np.array(total_x)
        total_y = np.array(total_y)

        # calculate mean and std over the training data 
        if train_valid == "train":
            mean = total_x.mean(axis=0)
            std = total_x.std(axis=0)
            np.save("train_mean_std_" + model_name + ".npy", [mean, std])
        else:
            try:
                mean_std = np.load("train_mean_std_" + model_name + ".npy")
                mean = mean_std[0]
                std = mean_std[1]
            except:
                print("mean, std not found")
                sys.exit()

        total_x_norm = (total_x - mean) / std
        total_x_norm = np.swapaxes(total_x_norm, 1, 2)
        print(total_x_norm)
        total_y = np.expand_dims(total_y, 2)

    # deal with single files during inference step
    else:
        audio_file = song
        x, y = load_single_xy(audio_file, mel_dir, label_dir)
        for i in range(0, x.shape[1] - RNN_INPUT_SIZE, 1):
            x_segment = x[:, i: i + RNN_INPUT_SIZE]
            y_segment = y[i: i + RNN_INPUT_SIZE]
            total_x.append(x_segment)
            total_y.append(y_segment)

        total_x = np.array(total_x)
        total_y = np.array(total_y)
        try:
            mean_std = np.load("train_mean_std_" + model_name + '.npy')
            mean = mean_std[0]
            std = mean_std[1]
        except:
            print("mean, std not found")
            sys.exit()

        total_x_norm = (total_x - mean) / std
        total_x_norm = np.swapaxes(total_x_norm, 1, 2)
        total_y = np.expand_dims(total_y, 2)

    return total_x_norm, total_y


# =========================== MedleyDB ================================== #

def load_label_mdb(audio_feature, label_file):
    ''' Process and load label for the given audio
    Args :
        audio_feature : computed hpss+melgram of audio. Shape=(80, total_frames) 
        audio_label_file : path to label file
    Return :
        frame_level_label : list of ground truth annotations per frame. Shape=(total_frames,)
    '''
    frame_level_label = np.zeros((audio_feature.shape[1],))
    sample_level_label = np.load(label_file)

    for i in range(len(frame_level_label)):
        t1 = librosa.frames_to_time(i, sr=SR, hop_length=N_HOP2)
        f2 = librosa.time_to_samples(t1, sr=22050)

        if f2 >= len(sample_level_label):
            continue
        frame_level_label[i] = sample_level_label[f2]

    return frame_level_label


def load_single_xy_mdb(audio_file, mel_dir, label_dir):
    ''' Create (audio feature, label) data pair for the givne audio

    Args : 
        audio_file : name of the audio feature npy file
        mel_dir : path to the folder containing npy files
        label_dir : path to the label folder 

    Return : 
        audio_feature : audio feature. Shape=(80, total_frames)
        label : annotation for the audio. Shape=(total_frames,)
    '''

    # load audio
    audio_feature = np.load(os.path.join(mel_dir, audio_file))  # (80, frame_len)
    print(audio_feature.shape)

    # load label 
    audio_file_name = audio_file.split('.')[0].replace("MIX", "RAW")
    audio_file_name = audio_file_name.replace("p0", "RAW")
    audio_file_name = audio_file_name.replace("p6", "RAW")
    audio_file_name = audio_file_name.replace("p12", "RAW")
    audio_file_name = audio_file_name.replace("m6", "RAW")
    audio_file_name = audio_file_name.replace("m12", "RAW")
    audio_file_name = os.path.join(label_dir, audio_file_name + '.npy')
    label = load_label_mdb(audio_feature, audio_file_name)

    print(audio_file, audio_feature.shape)
    return audio_feature, label


def load_xy_data_mdb(song, mel_dir, label_dir, model_name):
    ''' Load all x,y pair in the given dataset into a list of x_data and list of y_label which are segmented into RNN_INPUT_SIZE shape.

   Args:
       song : name of song 
       mel_dir: path to folder of the saved audio feature npy files
       label_dir : path to the folder of label files
       model_name: name of the current rnn model

   Return:
       total_x_norm: Shape=(1,218,80)
       total_y: (1, 218, 1)
   '''

    total_x = []
    total_y = []
    x, y = load_single_xy_mdb(song, mel_dir, label_dir)
    for i in range(0, x.shape[1] - RNN_INPUT_SIZE, 1):
        x_segment = x[:, i: i + RNN_INPUT_SIZE]
        y_segment = y[i: i + RNN_INPUT_SIZE]
        total_x.append(x_segment)
        total_y.append(y_segment)

    total_x = np.array(total_x)
    total_y = np.array(total_y)

    # normalize over training set 
    try:
        mean_std = np.load("train_mean_std_" + model_name + '.npy')
        mean = mean_std[0]
        std = mean_std[1]
        total_x = (total_x - mean) / std
    except:
        print("mean, std not found.")
        sys.exit()

    total_x = np.swapaxes(total_x, 1, 2)
    total_y = np.expand_dims(total_y, 2)

    return total_x, total_y
