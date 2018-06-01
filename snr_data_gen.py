import os
import sys
import librosa
import yaml
import numpy as np
import subprocess

''' SNR modified data generator for Experiment 2 '''

mixfilepath = '/media/bach3/dataset/MedleyDB/Audio/'  # path to MIX files
vocalfilepath = '/media/bach1/dataset/MedleyDB/voice_only/'  # path to RAW vocal files
list_of_vocal_files = os.listdir(vocalfilepath)
savepath = './loudness/'
audio_length = 15000
SR = 22050
START = 0
DURATION = 60


def rms(song):
    sq = np.square(song)
    meansq = np.mean(sq)
    rootmeansq = np.sqrt(meansq)
    return rootmeansq


def normalize(mix):
    norm = 1 / (np.max(abs(mix)))
    return norm


def make_mixfile(songtitle, start, duration):
    '''
    Args :
        songtitle :
        start :
        duration :
    '''
    yamlfilepath = os.path.join(mixfilepath, songtitle, songtitle + '_METADATA.yaml')

    inst_audios = []
    stemfolder = None
    with open(yamlfilepath, 'r') as stream:
        song_dict = yaml.load(stream)
        stemfolder = song_dict['stem_dir']
        for stem in (song_dict['stems']):
            if (song_dict['stems'][stem]['component'] == 'melody') and (
                    song_dict['stems'][stem]['instrument'] == 'male singer' or song_dict['stems'][stem][
                'instrument'] == 'female singer'):
                continue

            stemfile = song_dict['stems'][stem]['filename']
            print(stemfile)
            inst_audios.append(stemfile)

    mixsong = np.zeros((0,))
    for track in inst_audios:
        loaded_track, sr = librosa.load(os.path.join(mixfilepath, songtitle, stemfolder, track), sr=SR, offset=start,
                                        duration=duration, mono=True)
        if mixsong.shape[0] == 0:
            mixsong = loaded_track
        else:
            mixsong += loaded_track
            # mixsong = mixsong * rms(mixsong) * (1-1/np.power(2,16))
    mixsong = normalize(mixsong) * mixsong
    return mixsong


def change_decibel(vocalfile, level):
    vocsong = AudioSegment.from_wav(vocalfile)
    vocsong = vocsong + level

    return vocsong


if __name__ == '__main__':

    dB_scale = [-12, -6, 0, 6, 12]
    save_dir = ['voc_m12', 'voc_m6', 'voc_p0', 'voc_p6', 'voc_p12']

    for song in list_of_vocal_files:
        print(song)
        songname = song.split('_RAW')[0]
        print(songname)
        if song[0] == '.':
            continue

        # first make instrument mix file 
        vocalfile = os.path.join(vocalfilepath, song)

        voc_signal, _ = librosa.load(vocalfile, sr=SR, mono=True, offset=START, duration=DURATION)
        total_len = librosa.get_duration(voc_signal, sr=SR)  # in seconds
        print(total_len)
        # voc_signal *= 0.7
        voc_signal *= 0.7

        '''
        if total_len > 60:
            start = 30
        elif total_len > 30 :
            start = total_len - 30
        else:
        '''
        inst_mix = make_mixfile(songname, START, DURATION)
        inst_mix *= 0.8
        print(vocalfile)

        for i in range(len(dB_scale)):
            # modify audio and mix it as well
            R_dB = dB_scale[i]
            curr_savepath = os.path.join(savepath, save_dir[i], 'songs')
            if not os.path.isdir(curr_savepath):
                os.makedirs(curr_savepath)

            R = np.power(10, (R_dB / 20.0))  # db to amplitude conversion
            voc_to_inst_ratio = R

            print("ratio", voc_to_inst_ratio)

            # mix file 
            final_mix = voc_signal * voc_to_inst_ratio + inst_mix

            if R_dB < 0:
                lev = 'm' + str(abs(R_dB))
            else:
                lev = 'p' + str(abs(R_dB))

            librosa.output.write_wav(os.path.join(curr_savepath, songname + '_' + lev + '.wav'), final_mix, sr=SR)
