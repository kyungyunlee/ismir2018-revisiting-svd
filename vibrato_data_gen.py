import os
import numpy as np
import librosa
import itertools
from scipy import signal
from scipy.signal import butter, lfilter, freqz

''' Sawtooth generator for Stress Test 1 : Vibrato 
'''

order = 6
cutoff = 3.667

SR = 44100  # sampling rate


# BW = np.array([130, 70, 160])  # bandwidths for formants


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_modulation(f0, delta_f, rate, len_sec):
    """
    Args : 
        f0 : fundamental freq of the pure tone [Hz]
        delta_f : amplitude of the frequency oscilliation
        rate : [Hz] of the vibrato
        len_sec : length of the signal [second]
    Return :
        y : generated signal 
    """
    dt = 1. / 44100
    time = np.arange(0., len_sec, dt)
    frequency = f0 - delta_f * \
                     np.sin(2 * np.pi * time * rate)  # a 1Hz oscillation
    print("freq", frequency)

    phase_correction = np.add.accumulate(time * \
                                         np.concatenate((np.zeros(1), 2 * np.pi * (frequency[:-1] - frequency[1:]))))

    '''
    waveform = np.sin(2 * np.pi * time * frequency + phase_correction)
    '''
    waveform = signal.sawtooth(2 * np.pi * time * frequency + phase_correction)
    y = butter_lowpass_filter(waveform, cutoff, frequency[0], order)
    y *= 0.5
    print(type(y))
    return y


def gen_speech(F, bw, sig, fs):
    """
    Args : 
        F: formant frequencies (np array)
        sig: original signal to filter
        fs: sampling frequency [Hz]
    """
    nsecs = len(F)
    R = np.exp(-np.pi * bw / fs)  # pope radii
    theta = 2 * np.pi * F / fs  # pole angles
    poles = R * np.exp(1j * theta)

    A = np.real(np.poly(np.concatenate([poles, poles.conj()], axis=0)))
    B = np.zeros(A.shape)
    B[0] = 1
    r, p, f = signal.residuez(B, A)
    As = np.zeros((nsecs, 3), dtype=np.complex)
    Bs = np.zeros((nsecs, 3), dtype=np.complex)

    for idx, i in enumerate(range(1, 2 * nsecs + 1, 2)):
        j = i - 1

        Bs[idx] = [r[j] + r[j + 1], -(r[j] * p[j + 1] + r[j + 1] * p[j]), 0]
        As[idx] = [1, -(p[j] + p[j + 1]), p[j] * p[j + 1]]

    sos = np.concatenate([As, Bs], axis=1)
    iperr = np.abs(np.imag(sos)) / (np.abs(sos) + 1e-10)
    sos = np.real(sos)
    Bh, Ah = signal.sos2tf(sos)
    nfft = 512

    H = np.zeros((nsecs + 1, nfft))
    for i in range(nsecs):
        Hiw, w = signal.freqz(Bs[i, :], As[i, :])
        H[i + 1, :] = np.conj(Hiw[:])

    H[0, :] = np.sum(H[1:, :], axis=0)

    speech = signal.lfilter([1], A, sig)
    speech = speech - speech.mean()
    speech = speech / np.max(np.abs(speech))
    return speech


if __name__ == "__main__":
    f0 = 220  # fundamental freq of the signal
    len_sec = 4  # length of the signal [second]

    try:
        os.mkdir('sawtooth_200')
    except:
        pass

    try:
        os.mkdir('sawtooth_200/songs')
    except:
        pass

    # https://soundbridge.io/formants-vowel-sounds
    Fs = np.array([[800, 1150, 2800],
                   [400, 1600, 2700],
                   [350, 1700, 2700],
                   [450, 800, 2830],
                   [325, 700, 2530]])
    BWs = np.array([[80, 90, 120],
                    [60, 80, 120],
                    [50, 100, 120],
                    [70, 80, 100],
                    [50, 60, 170]])
    speech_names = ['a', 'e', 'i', 'o', 'u']

    semitones = [0.01, 0.1, 0.3, 0.6, 1, 2, 4, 8]  # amount of vibrato [semitones]

    delta_fs = [np.round(f0 * 2 ** (smt / 12.) - f0) for smt in semitones]  # amount of vibrato [Hz]
    print(delta_fs)
    rates = [0.5, 1, 2, 4, 6, 8, 10]  # how fast is the vibrato [num_vibrato per second]
    for delta_f, rate in itertools.product(delta_fs, rates):
        print(delta_f, rate)
        y = get_modulation(f0, delta_f, rate, len_sec)
        filename = 'sawtooth_200/songs/' + 'modulated_%d_%d_%d.wav' % (
            int(f0), int(delta_f), int(rate))
        librosa.output.write_wav(filename, y=y, sr=SR)

        for F, BW, speech_name in zip(Fs, BWs, speech_names):
            y_modi = gen_speech(F, BW, y, SR)
            filename = 'sawtooth_200/songs/' + 'modulated_%s_%d_%d_%d.wav' % (
                speech_name, int(f0), int(delta_f), int(rate))
            librosa.output.write_wav(filename, y=y_modi, sr=SR)
