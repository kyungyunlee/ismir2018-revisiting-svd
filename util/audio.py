'''
Audio processing functions 
'''
import os
import numpy as np
import librosa


def hpss(y, n_fft, hop_length):
    ''' Compute harmonic-percussive source separation with librosa  
    Args :
        y : audio signal. Shape=(total_frames, )
        n_fft : fft window size in frames
        hop_length : hop length in frames
    Return :
        y_harm : harmonic component of audio signal. Shape=(total_frames,)
        y_perc : percussive component of audio signal. Shape=(total_frames, )
    '''
    # compute stft
    stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
    # compute hpss
    stft_harm, stft_perc = librosa.decompose.hpss(stft)
    # inverse stft 
    y_harm = librosa.util.fix_length(librosa.core.istft(stft_harm, hop_length=hop_length, dtype=y.dtype), len(y))
    y_perc = librosa.util.fix_length(librosa.core.istft(stft_perc, hop_length=hop_length, dtype=y.dtype), len(y))

    return y_harm, y_perc


def ono_hpss(y, n_fft, hop_length):
    ''' HPSS implementation from Ono
    '''
    gamma = 1  # 0.3 for ranged compressed version as suggested in the paper but doesntsound good
    alpha = 0.3
    max_iter = 50

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    print("stft", stft.shape)
    _, phase = librosa.magphase(stft)
    power_spec = (np.abs(stft)) ** (2 * gamma)
    f_len, t_len = power_spec.shape
    # set initial values of H, P
    H = power_spec / 2
    P = power_spec / 2

    # (frequency,time)
    for k in range(max_iter):
        H_left = np.zeros((f_len, t_len))
        H_right = np.zeros((f_len, t_len))
        H_left[:, :t_len - 1] = H[:, 1:]
        H_right[:, 1:] = H[:, :t_len - 1]

        P_up = np.zeros((f_len, t_len))
        P_down = np.zeros((f_len, t_len))
        P_up[:f_len - 1, :] = P[1:, :]
        P_down[1:, :] = P[:f_len - 1, :]

        delta = alpha * (H_left - 2 * H + H_right) / 4 - ((1 - alpha) * (P_up - 2 * P + P_down) / 4)
        H = np.minimum(np.maximum(H + delta, 0), power_spec)
        P = power_spec - H

    # mask
    perc_mask = H < P
    percussive = np.multiply(perc_mask, power_spec)
    harmonic = power_spec - percussive

    # inverse stft
    p_out = librosa.util.fix_length(
        librosa.core.istft(percussive ** (1 / 2 * gamma) * phase, hop_length=hop_length, dtype=y.dtype), len(y))
    h_out = librosa.util.fix_length(
        librosa.core.istft(harmonic ** (1 / 2 * gamma) * phase, hop_length=hop_length, dtype=y.dtype), len(y))
    print(p_out, h_out)

    return h_out, p_out


def log_melgram(sig, sr, n_fft, hop_length, n_mels, fmin=0.0, fmax=None):
    ''' Compute log of melspectrogram
    Args : 
        sig : 1-d audio signal
        n_fft : fft window size in frames
        hop_length : hop_length in frames
        n_mels : number of mel bands
    Return :
        log_melspec : logarithmized melspectrogram
    '''
    melspec = librosa.feature.melspectrogram(sig, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin,
                                             fmax=fmax, power=1.0)
    log_melspec = librosa.amplitude_to_db(melspec)
    return log_melspec
