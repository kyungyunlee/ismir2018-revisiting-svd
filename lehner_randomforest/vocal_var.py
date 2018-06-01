''' Features from Lehner et al.2015 paper are computed with matlab as provided by the authors. Here, vocal variance is computed and concatenate with the other features from matlab code. (I will not be uploading the matlab code as it is not mine, so please ask someone with it ;)) 
'''
import os
import numpy as np
from numpy import ma
import librosa
import argparse
import scipy.io
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

# vocal variance parameters 
FRAMESIZE = 2205
HOPSIZE = 441


def concat_feat(audio_path, set_dir, matlab_dir, feat_save_dir):
    if not os.path.exists(feat_save_dir):
        os.makedirs(feat_save_dir)

    for set_name in set_dir:
        list_of_files = os.listdir(os.path.join(audio_path, set_name))
        for f in list_of_files:
            if f[0] == '.':
                continue
            audiofile = os.path.join(audio_path, set_name, f)
            y, _ = librosa.load(audiofile, sr=SR)
            vv = vocal_var(y)

            audio_mat = scipy.io.loadmat(os.path.join(matlab_dir, set_name, f.split('.')[0] + '.mat'))
            audio_feat = audio_mat['all_features']

            vv_len = vv.shape[0]
            other_len = audio_feat.shape[0]

            final_len = min(vv_len, other_len)

            final_feat = np.concatenate((audio_feat[:final_len, :], vv[:final_len, :]), axis=1)

            print(feat_save_dir + f.split('.')[0] + '.npy')
            if not os.path.exists(os.path.join(feat_save_dir, set_name)):
                os.makedirs(os.path.join(feat_save_dir, set_name))
            np.save(os.path.join(feat_save_dir, set_name, f.split('.')[0] + '.npy'), final_feat)


def vocal_var(audiofile):
    ''' Compute vocal variance
    '''
    mfcc = librosa.feature.mfcc(audiofile, sr=SR, n_mfcc=30, n_fft=FRAMESIZE, hop_length=HOPSIZE, n_mels=128)
    mfcc = mfcc.swapaxes(0, 1)
    vv = np.empty([len(mfcc), 5])
    for i in range(len(mfcc)):
        for j in range(5):
            vv[i][j] = np.var(mfcc[max(0, i - 5): min(len(mfcc), i + 6), j + 1])
    return vv


if __name__ == '__main__':
    if args.dataset == 'jamendo':
        set_dir = ['train', 'valid', 'test']
        concat_feat(JAMENDO_DIR, set_dir, MATLAB_JAMENDO_DIR, FEAT_JAMENDO_DIR)
    elif args.dataset == 'vibrato':
        concat_feat(SAW_DIR, [''], MATLAB_SAW_DIR, FEAT_SAW_DIR)
    elif args.dataet == 'snr':
        dblevel = ['voc_p0', 'voc_p6', 'voc_p12', 'voc_m6', 'voc_m12']
        for db in dblevel:
            LD = '/media/bach4/kylee/loudness/' + db + '/songs/'
            MLD = '/media/bach4/kylee/loudness/' + db + '/randomforest_matlab/'
            FLD = '/media/bach4/kylee/loudness/' + db + '/randomforest_feat_dir/'
            concat_feat(LD, [''], MLD, FLD)
