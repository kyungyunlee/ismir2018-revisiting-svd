''' Functions for loading data for training and testing '''
import os
import sys
import numpy as np
import librosa
from config_cnn import *
from scipy.signal import medfilt


# ======================== JAMENDO & OTHERS  ============================ #

def load_label(audio_spec, audio_label_file):
    ''' Process and load label for the given audio
    Args:
        audio_spec : melgram of the audio. Shape=(n_bins, total_frames) ex.(80,14911)
        audio_label_file : path to the label file ex. './jamendo/jamendo_lab/02 - The Louis...lab' 
    Return :
        lab : list of ground truth annotations per frame. Shape=(total_frames, )
    '''
    with open(audio_label_file, 'r') as f:
        total_frames = audio_spec.shape[1]
        label = np.zeros((total_frames,))

        for line in f:
            l = line.strip('\n').split(' ')
            start = librosa.time_to_frames(float(l[0]), sr=SR, hop_length=HOP_LENGTH)
            end = librosa.time_to_frames(float(l[1]), sr=SR, hop_length=HOP_LENGTH)

            is_vocal = 1 if l[2] == 'sing' or l[2] == '1' else 0
            label[start:end] = int(is_vocal)

    return label


def load_single_xy(audio_file, mel_dir, label_dir):
    ''' Create (melgram, label) data pair for the given audio
    
    Args:
        audio_file : name of the melgram npy file 
        mel_dir : path to the folder containing npy files
        label_dir : path to the lab folder

    Return :
        audio_mel_feature : melgram of the audio. Shape=(80, total_frames)
        label : annotation for the audio. Shape=(total_frames,)
    '''
    # load audio    
    if mel_dir == MEL_JAMENDO_DIR:
        mel_dir = os.path.join(mel_dir, 'test')
    audio_mel_feature = np.load(os.path.join(mel_dir, audio_file))  # (80, frame_len)

    # load label 
    if label_dir == '':
        label = np.zeros((audio_mel_feature.shape[1]))
    else:
        audio_file_name = audio_file.split('.')[0]
        audio_label_file = os.path.join(label_dir, audio_file_name + '.lab')
        label = load_label(audio_mel_feature, audio_label_file)

    print(audio_file, audio_mel_feature.shape)
    return audio_mel_feature, label


def load_xy_data(song, mel_dir, label_dir, model_name, train_valid=None):
    ''' Load all x,y pair into a list of x_data and list of y_labels which are segmented into CNN_INPUT_SIZE shape.

    Args: 
        song : name of the test song npy file. Set to None if train_valid =='train' or 'valid'   
        mel_dir : path to folder of saved melgram npy files
        label_dir : path to folder of label files
        model_name : name of the current cnn model
        train_valid : 'train' or 'valid' or none 
    Return:
        total_x_norm : Shape=(total_frames, n_bins, input_frame_size) ex.(166,80,115 
        total_y : Shape=(total_frames,)
    '''
    train_valid_set = ['train', 'valid']

    total_x = []
    total_y = []

    # if in train_valid_set, compute and save mean && std over the training data 
    if train_valid in train_valid_set:
        for audio_file in os.listdir(os.path.join(mel_dir, train_valid)):
            x, y = load_single_xy(audio_file, os.path.join(mel_dir, train_valid), label_dir)
            # segment into 115 frames
            for i in range(0, x.shape[1] - CNN_INPUT_SIZE, CNN_OVERLAP):
                x_segment = x[:, i: i + CNN_INPUT_SIZE]
                # pick the center frame label 
                y_label = y[i + int(CNN_INPUT_SIZE // 2) + 1]
                total_x.append(x_segment)
                total_y.append(y_label)

        assert len(total_x) == len(total_y)

        total_x = np.array(total_x)
        total_y = np.array(total_y)

        # calculate mean and std to normalize
        if train_valid == "train":
            mean = total_x.mean(axis=0)
            std = total_x.std(axis=0)
            np.save("train_mean_std_" + model_name + ".npy", [mean, std])
        elif train_valid == 'valid':
            try:
                mean_std = np.load("train_mean_std_" + model_name + ".npy")
                mean = mean_std[0]
                std = mean_std[1]
            except:
                print("mean, std not found.")
                sys.exit()

        total_x_norm = (total_x - mean) / std
        total_x_norm = np.expand_dims(total_x_norm, axis=3)
        print(total_x_norm.shape)

    else:
        audio_file = song
        x, y = load_single_xy(audio_file, mel_dir, label_dir)
        for i in range(0, x.shape[1] - CNN_INPUT_SIZE, 1):
            x_segment = x[:, i: i + CNN_INPUT_SIZE]
            # pick the center frame label 
            y_label = y[i + int(CNN_INPUT_SIZE // 2) + 1]
            total_x.append(x_segment)
            total_y.append(y_label)
        total_x = np.array(total_x)
        total_y = np.array(total_y)

        # normalize with training mean, std
        try:
            mean_std = np.load("train_mean_std_" + model_name + ".npy")
            mean = mean_std[0]
            std = mean_std[1]
        except:
            print("mean, std not found")
            sys.exit()

        total_x_norm = (total_x - mean) / std
        total_x_norm = np.expand_dims(total_x_norm, axis=3)

    return total_x_norm, total_y


# ======================== MEDLEYDB ============================ #
''' A different data loading function for medlydb'''


def load_label_mdb(audio_mel_feature, label_file):
    ''' Load annotation from medleydb melody annotation file using pitch > 0.0
    Args :
        audio_mel_feature : melgram of audio. Shape=(n_bins, total_frames) ex.(80,10286)
        label_file : path to the label file
    Return :
        frame_level_label: frame level vocal segment annotation for the audio. Shape=(total_frames,)
    '''
    frame_level_label = np.zeros((audio_mel_feature.shape[1],))
    sample_level_label = np.load(label_file)

    for i in range(len(frame_level_label)):
        t1 = librosa.frames_to_time(i, sr=SR, hop_length=HOP_LENGTH)
        f2 = librosa.time_to_samples(t1, sr=22050)

        if f2 >= len(sample_level_label):
            continue
        frame_level_label[i] = sample_level_label[f2]

    return frame_level_label


def load_single_xy_mdb(audio_file, mel_dir, label_dir):
    '''Create (audio, label) data pair for the given audio file name

    Args:
        audio_file : name of the melgram npy file 
        mel_dir : path to the folder containing npy files
        label_dir : path to the label folder

    Return :
        audio_mel_feature : melgram of the audio. Shape=(80, total_frames)
        label : annotation for the audio. Shape=(total_frames,)
    '''
    # load audio    
    audio_mel_feature = np.load(os.path.join(mel_dir, audio_file))  # (80, frame_len)

    # load label 
    audio_label_file = audio_file.split('.')[0].replace("MIX", "RAW")
    audio_label_file = audio_label_file.replace("p0", "RAW")
    audio_label_file = audio_label_file.replace("p6", "RAW")
    audio_label_file = audio_label_file.replace("p12", "RAW")
    audio_label_file = audio_label_file.replace("m6", "RAW")
    audio_label_file = audio_label_file.replace("m12", "RAW")
    print(audio_label_file)
    audio_label_file = os.path.join(label_dir, audio_label_file + '.npy')
    label = load_label_mdb(audio_mel_feature, audio_label_file)

    print(audio_file, audio_mel_feature.shape)
    return audio_mel_feature, label


def load_xy_data_mdb(song, mel_dir, label_dir, model_name):
    ''' Load all x,y pair in the given dataset into a list of x_data and list of y_labels

    Args: 
        song : name of the song  
        mel_dir : path to folder of saved melgram npy files
        label_dir : path to folder of label files
        model_name : name of the current cnn model
    Return:
        total_x_norm : Shape=(total_frames, n_bins, input_frame_size) ex.(166,80,115 
        total_y : Shape=(total_frames,)
    '''

    total_x = []
    total_y = []
    x, y = load_single_xy_mdb(song, mel_dir, label_dir)
    # segment into 115 frames
    for i in range(0, x.shape[1] - CNN_INPUT_SIZE, 1):
        x_segment = x[:, i: i + CNN_INPUT_SIZE]
        # pick the center frame label 
        y_label = y[i + int(CNN_INPUT_SIZE // 2)]
        total_x.append(x_segment)
        total_y.append(y_label)

    assert len(total_x) == len(total_y)
    total_x = np.array(total_x)
    total_y = np.array(total_y)

    # normalize with the training mean, std
    try:
        mean_std = np.load("train_mean_std_" + model_name + ".npy")
        mean = mean_std[0]
        std = mean_std[1]
    except:
        print("mean, std not found")
        sys.exit()

    total_x = (total_x - mean) / std
    total_x = np.expand_dims(total_x, axis=3)

    return total_x, total_y
