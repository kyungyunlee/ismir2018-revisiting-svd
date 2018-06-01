import os
import sys
import numpy as np
import librosa
import argparse
from config_cnn import *

sys.path.append('../util')
from audio import log_melgram

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


def compute_and_save_melspectrogram(base_dir, mel_dir, set_list=['']):
    ''' Given a directory containing audio files, compute melgrams and save to another directory
    Args :
        base_dir : Path to the folder containing audio files or containing train/valid/test set folders ex.'../jamendo/'
        mel_dir : Path to folder where melgram computed npy files will be saved
        set_list : if applicable, list of sub folders that contain audio files ex. ['train', 'valid', 'test'] 

    Return : 
        None
    '''
    for set_dir in set_list:
        mel_set_dir = os.path.join(mel_dir, set_dir)
        if not os.path.exists(mel_set_dir):
            os.makedirs(mel_set_dir)

        for audio_file in os.listdir(os.path.join(base_dir, set_dir)):
            print(audio_file)
            if audio_file[0] == '.':
                continue

            audio_file_name = audio_file.split('.')[0]
            y, _ = librosa.load(os.path.join(base_dir, set_dir, audio_file), sr=SR)
            mel_audio = log_melgram(y, SR, FRAME_LEN, HOP_LENGTH, N_MELS, 27.5, 8000)
            np.save(os.path.join(mel_set_dir, audio_file_name + '.npy'), mel_audio)


if __name__ == '__main__':
    if args.dataset == 'jamendo':
        set_list = ['train', 'valid', 'test']
        compute_and_save_melspectrogram(JAMENDO_DIR, MEL_JAMENDO_DIR, set_list)
    elif args.dataset == 'vibrato':
        compute_and_save_melspectrogram(SAW_DIR, MEL_SAW_DIR, [''])

    elif args.dataset == 'snr':
        dblevel = ['voc_p0', 'voc_p6', 'voc_p12', 'voc_m6', 'voc_m12']
        for db in dblevel:
            LD = '../loudness/' + db + '/songs/'
            MLD = '../loudness/' + db + '/schluter_mel_dir/'

            compute_and_save_melspectrogram(LD, MLD, [''])
