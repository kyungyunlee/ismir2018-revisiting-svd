import os
import numpy as np
import pickle
import keras
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.models import load_model
from scipy.ndimage.filters import median_filter
import argparse
from load_data import *
from model import *
from config_cnn import *

np.random.seed(0)
# set gpu number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


def test(model_name, test_set, song=None):
    ''' Test the model!
    Args:
        model_name : name of the model ex.20180331
        test_set : name of the data set ex. 'jamendo', 'vibrato'

    Return : 
        y_pred_cont : Non thresholded prediction values per frame. Shape = (total_frames, n_bins, input_frame_size) ex. (2000, 80, 115)
        y_pred : Thresholded prediction values per frame. Shape = (total_frames, n_bins, input_frame_size) 
        y_test : Ground truth values per frame. Shape = (total_frames, n_bins, input_frame_size) 
    '''

    if test_set == 'test':
        x_test, y_test = load_xy_data(song, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, model_name)
    elif test_set == 'vibrato':
        x_test, y_test = load_xy_data(song, MEL_SAW_DIR, '', model_name)
    elif test_set[:3] == 'voc':
        MLD = '../loudness/' + test_set + '/schluter_mel_dir/'
        x_test, y_test = load_xy_data_mdb(song, MLD, MDB_LABEL_DIR, model_name)

    # predict
    y_pred = loaded_model.predict(x_test, verbose=1)
    y_pred_cont = y_pred.copy()
    y_pred = y_pred.reshape(-1)

    y_test[y_test >= THRESHOLD] = True
    y_test[y_test < THRESHOLD] = False
    y_test = y_test.astype(int)

    y_pred[y_pred >= THRESHOLD] = True
    y_pred[y_pred < THRESHOLD] = False
    y_pred = y_pred.astype(int)

    # postprocessing : median filtering 
    # y_pred = median_filter(y_pred, 56, mode='nearest')

    accuracy_each = (((y_test.shape[0] - np.sum(np.abs(y_pred - y_test)))) / float(y_test.shape[0])) * 100.0
    accuracy = np.mean(accuracy_each)

    f1 = f1_score(y_test, y_pred, average='binary')
    pr = precision_score(y_test, y_pred, average='binary')
    re = recall_score(y_test, y_pred, average='binary')

    print('Accuracy: ', accuracy)
    print('Precision: ', pr)
    print('Recall: ', re)
    print('F1-score: ', f1)
    return y_pred_cont, y_pred, y_test


if __name__ == '__main__':
    # best : weights/cnn_20180531-1
    if args.dataset == 'jamendo':
        test_sets = ['jamendo']
    elif args.dataset == 'vibrato':
        test_sets = ['vibrato']
    elif args.dataset == 'snr':
        test_sets = ['voc_p0', 'voc_p6', 'voc_m6', 'voc_p12', 'voc_m12']
    else:
        print("unknown dataset")
        sys.exit()

    # load model 
    model_name = args.model_name
    loaded_model = load_model('weights/cnn_' + model_name + '.h5')
    print("loaded model")
    print(loaded_model.summary())

    ''' Save the predictions to pkl file for model analysis 
    pkl file : contains dictionary of {'songname' : [y_pred_cont, y_pred, y_test]}
    '''
    for test_set in test_sets:
        predicted_values = {}
        y_preds = []
        y_tests = []

        if test_set == 'jamendo':
            list_of_songs = os.listdir(MEL_JAMENDO_DIR + 'test')
            test_set = 'test'
        elif test_set == 'vibrato':  # vibrato test
            list_of_songs = os.listdir(MEL_SAW_DIR)
        elif test_set[:3] == 'voc':  # SNR test
            MLD = '../loudness/' + test_set + '/schluter_mel_dir/'
            list_of_songs = os.listdir(MLD)

        for song in list_of_songs:
            y_pred_cont, y_pred, y_test = test(model_name, test_set, song=song)
            for i in range(len(y_pred)):
                y_preds.append(y_pred[i])
                y_tests.append(y_test[i])
            print(y_pred, y_test)

            # pad front and last of the predicted value list
            ones = np.ones((CNN_INPUT_SIZE // 2,))
            zeros = np.zeros((CNN_INPUT_SIZE // 2,))
            pred_cont_pad_front = zeros + y_pred_cont[0]
            pred_cont_pad_end = zeros + y_pred_cont[-1]
            pred_pad_front = ones if y_pred[0] else zeros
            pred_pad_end = ones if y_pred[-1] else zeros
            test_pad_front = ones if y_test[0] else zeros
            test_pad_end = ones if y_test[-1] else zeros

            y_pred_cont = np.append(pred_cont_pad_front, y_pred_cont)
            y_pred_cont = np.append(y_pred_cont, pred_cont_pad_end)
            y_pred = np.append(pred_pad_front, y_pred)
            y_pred = np.append(y_pred, pred_pad_end)
            y_test = np.append(test_pad_front, y_test)
            y_test = np.append(y_test, test_pad_end)
            predicted_values[song] = [y_pred_cont, y_pred, y_test]

        if test_set == 'test':
            test_set = 'jamendo'
        pickle.dump(predicted_values, open(test_set + '.pkl', 'wb'))

        y_preds = np.array(y_preds)
        y_tests = np.array(y_tests)

        # calculate test score
        acc = (len(y_tests) - np.sum(np.abs(y_preds - y_tests))) / float(len(y_tests))
        f1 = f1_score(y_tests, y_preds, average='binary')
        pr = precision_score(y_tests, y_preds, average='binary')
        re = recall_score(y_tests, y_preds, average='binary')

        tn, fp, fn, tp = confusion_matrix(y_tests, y_preds).ravel()
        fp_rate = fp / (fp + tn)
        fn_rate = fn / (fn + tp)

        print("TEST SCORES")
        print('Accuracy: ', acc)
        print('Precision: ', pr)
        print('Recall: ', re)
        print('F1-score: ', f1)
        print('fp rate', fp_rate, 'fn_rate', fn_rate)
