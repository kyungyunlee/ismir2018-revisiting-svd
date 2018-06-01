''' Preprocessing audio files to mel features '''
import os
import sys
import numpy as np
import librosa
import argparse
from config_rnn import *

sys.path.append('../util')
from audio import ono_hpss, log_melgram

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


def process_single_audio(audio_file):
    ''' Compute double stage HPSS for the given audio file
    Args : 
        audio_file : path to audio file 
    Return :
        mel_D2_total : concatenated melspectrogram of percussive, harmonic components of double stage HPSS. Shape=(2 * n_bins, total_frames) ex. (80, 2004) 
    '''
    audio_src, _ = librosa.load(audio_file, sr=SR)
    # Normalize audio signal
    audio_src = librosa.util.normalize(audio_src)
    # first HPSS
    D_harmonic, D_percussive = ono_hpss(audio_src, N_FFT1, N_HOP1)
    # second HPSS
    D2_harmonic, D2_percussive = ono_hpss(D_percussive, N_FFT2, N_HOP2)

    assert D2_harmonic.shape == D2_percussive.shape
    print(D2_harmonic.shape, D2_percussive.shape)

    # compute melgram 
    mel_harmonic = log_melgram(D2_harmonic, SR, N_FFT2, N_HOP2, N_MELS)
    mel_percussive = log_melgram(D2_percussive, SR, N_FFT2, N_HOP2, N_MELS)
    # concat
    mel_total = np.vstack((mel_harmonic, mel_percussive))

    print(mel_total.shape)
    return mel_total


def process_to_mel_feature(base_dir, mel_dir, set_list=['']):
    ''' Process audio files with double stage HPSS and save each to npy file
    Args : 
        base_dir : path to audio files
        mel_dir : path to save each HPSS computed npy files
        set_list : if applicable, ['train', 'valid', 'test'] else ['']
    Return :
        None
    '''

    for set_dir in set_list:
        mel_set_dir = os.path.join(mel_dir, set_dir)
        if not os.path.exists(mel_set_dir):
            os.makedirs(mel_set_dir)

        # audio processing 
        for audio_file in os.listdir(os.path.join(base_dir, set_dir)):
            print(audio_file)
            if audio_file[0] == '.':
                continue

            audio_file_name = audio_file.split('.')[0]
            audio_file_path = os.path.join(base_dir, set_dir, audio_file)
            norm_mel_D2_total = process_single_audio(audio_file_path)
            np.save(os.path.join(mel_set_dir, audio_file_name + '.npy'), norm_mel_D2_total)


if __name__ == '__main__':
    if args.dataset == 'jamendo':
        set_list = ['train', 'test', 'valid']
        process_to_mel_feature(JAMENDO_DIR, MEL_JAMENDO_DIR, set_list)
    elif args.dataset == 'vibrato':
        process_to_mel_feature(VIB_DIR, MEL_VIB_DIR, [''])
    elif args.dataset == 'snr':
        dblevels = ['voc_p0', 'voc_p6', 'voc_p12', 'voc_m6', 'voc_m12']
        for db in dblevels:
            LD = '../loudness/' + db + '/songs/'
            MLD = '../loudness/' + db + '/leglaive_mel_dir/'

            process_to_mel_feature(LD, MLD, [''])
