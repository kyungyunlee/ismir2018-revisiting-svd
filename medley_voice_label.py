''' 
Annotate MedleyDB vocal containing songs with the provided instrument activation annotation. 
Labels are at sample level with SR = 22050. 
'''
import os
import sys
import numpy as np
import librosa
import yaml
from scipy.signal import medfilt

base_dir = '/media/bach1/dataset/MedleyDB/voice_only/'
activation_dir = '/media/bach3/dataset/MedleyDB/Annotations/Instrument_Activations/ACTIVATION_CONF/'  # path to MedleyDB 'ACTIVATION_CONF' directory
metadata_dir = '/media/bach3/dataset/MedleyDB/Audio/'  # path to MedleyDB 'Audio' directory
label_dir = './mdb_voc_label/'

if not os.path.exists(label_dir):
    os.mkdir(label_dir)
vocal_list = open('./medleydb_vocal_songs.txt', 'r').readlines()

vocals = ['vocalists', 'female singer', 'male singer']

for voice_file in vocal_list:
    song_name = voice_file.strip('\n').replace('_RAW.wav', '')
    print(song_name)

    # find which stem number is the vocal stem 
    yaml_file = os.path.join(metadata_dir, song_name, song_name + '_METADATA.yaml')
    vocal_stem_num = ''
    with open(yaml_file, 'r') as stream:
        metadata = yaml.load(stream)
        print(metadata['mix_filename'])

        num_stems = len(metadata['stems'])
        for s in (list(metadata['stems'])):
            stem = (metadata['stems'][s])
            if stem['component'] == 'melody':
                if stem['instrument'] in vocals:
                    vocal_stem_num = s

        if vocal_stem_num == '':
            for s in (list(metadata['stems'])):
                if stem['component'] == '':
                    if stem['instrument'] in vocals:
                        vocal_stem_num = s

    assert vocal_stem_num != ''
    print(vocal_stem_num)

    vocal_stem = int(vocal_stem_num.replace('S', ''))
    activation_file = os.path.join(activation_dir, song_name + '_ACTIVATION_CONF.lab')

    # get length of the audio file 
    sig, sr = librosa.load(os.path.join(base_dir, song_name + '_RAW.wav'), sr=22050)
    label = np.zeros((len(sig),))

    # read the instrument activation file and get time and activations 
    f = open(activation_file, 'r').readlines()[1:]
    for i, line in enumerate(f[:-1]):
        line = line.strip('\n').split(',')
        start_time = float(line[0])
        end_time = float(f[i + 1].strip('\n').split(',')[0])
        start_sample = librosa.time_to_samples(start_time, sr=22050)
        end_sample = librosa.time_to_samples(end_time, sr=22050)
        activation = 1 if float(line[vocal_stem]) >= 0.5 else 0
        label[int(start_sample):int(end_sample)] = activation

    # fill the end of the label file 
    last_line = f[-1].strip('\n').split(',')
    last_start_sample = librosa.time_to_samples(float(last_line[0]), sr=22050)
    last_activation = 1 if float(last_line[vocal_stem]) >= 0.5 else 0
    label[int(last_start_sample):] = last_activation

    np.save(os.path.join(label_dir, song_name + '_RAW.npy'), label)

    ''' This code is for using energy level in RAW vocal tracks to create annotation. 
    base_dir = '/media/bach1/dataset/MedleyDB/voice_only/' # path to medleydb vocal stem files 
    threshold = 0.005 # signal amplitude threshold 
    sig, sr = librosa.load(os.path.join(base_dir, voice_file))

    label = np.abs(sig)
    label[label <= threshold] = 0
    label[label > threshold] = 1
    label = medfilt(label, 15)
    print (label.shape, np.count_nonzero(label))

    np.save(os.path.join(label_dir, voice_file.strip().split('.')[-2] + '.npy'), label) 
    '''
