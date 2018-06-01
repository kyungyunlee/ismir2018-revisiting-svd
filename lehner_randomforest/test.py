import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from scipy.ndimage.filters import median_filter
import argparse
from load_data import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()


def evaluate(model, test_set, song=None):
    if test_set == 'test':
        X_test, y_test = load_xy_data(song, FEAT_JAMENDO_DIR + test_set, JAMENDO_LABEL_DIR)
    elif test_set == 'vibrato':
        X_test, y_test = load_xy_data(song, FEAT_SAW_DIR, '')
    elif test_set[:3] == 'voc':
        MLD = '../loudness/' + test_set + '/randomforest_feat_dir/'
        X_test, y_test = load_xy_data_mdb(song, MLD, MDB_LABEL_DIR)

    X_test = X_test.reshape((X_test.shape[0], -1))
    y_pred = model.predict(X_test)
    result = model.score(X_test, y_test)

    # y_pred = median_filter(y_pred, 40, mode='nearest')
    f1 = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')

    print("f1", np.mean(f1), "precision", np.mean(precision), "recall", np.mean(recall))

    return y_pred, y_test


if __name__ == '__main__':
    # best model 20180531-1.sav
    if args.dataset == 'jamendo':
        test_sets = ['jamendo']
    elif args.dataset == 'vibrato':
        test_sets = ['vibrato']
    elif args.dataset == 'snr':
        test_sets = ['voc_m12', 'voc_m6', 'voc_p0', 'voc_p6', 'voc_p12']

    savefile = './weights/' + args.model_name + '.sav'
    loaded_model = pickle.load(open(savefile, 'rb'))

    for test_set in test_sets:
        predicted_values = {}
        y_preds = []
        y_tests = []

        if test_set == 'jamendo':
            list_of_songs = os.listdir(FEAT_JAMENDO_DIR + 'test')
            test_set = 'test'
        # for vibrato testing
        elif test_set == 'vibrato':
            list_of_songs = os.listdir(FEAT_SAW_DIR)
        # for SNR testing 
        elif test_set[:3] == 'voc':
            MLD = '../loudness/' + test_set + '/randomforest_feat_dir/'
            list_of_songs = os.listdir(MLD)

        for song in list_of_songs:
            y_pred, y_test = evaluate(loaded_model, test_set, song)
            for i in range(len(y_pred)):
                y_preds.append(y_pred[i])
                y_tests.append(y_test[i])

            # pad
            ones = np.ones((INPUT_FRAME_LEN // 2,))
            zeros = np.zeros((INPUT_FRAME_LEN // 2,))
            pred_pad_front = ones if y_pred[0] else zeros
            pred_pad_end = ones if y_pred[-1] else zeros
            test_pad_front = ones if y_test[0] else zeros
            test_pad_end = ones if y_test[-1] else zeros
            y_pred = np.append(pred_pad_front, y_pred)
            y_pred = np.append(y_pred, pred_pad_end)
            y_test = np.append(test_pad_front, y_test)
            y_test = np.append(y_test, test_pad_end)
            print(np.count_nonzero(y_test), np.shape(y_test))

            predicted_values[song] = [y_pred, y_test]
        pickle.dump(predicted_values, open(test_set + '.pkl', 'wb'))

        y_preds = np.array(y_preds)
        y_tests = np.array(y_tests)

        # calculate scores 
        acc = (len(y_tests) - np.sum(np.abs(y_preds - y_tests))) / float(len(y_tests))
        f1s = f1_score(y_tests, y_preds, average='binary')
        precisions = precision_score(y_tests, y_preds, average='binary')
        recalls = recall_score(y_tests, y_preds, average='binary')

        tn, fp, fn, tp = confusion_matrix(y_tests, y_preds).ravel()
        acc_confusion = (tp + tn) / (tp + fp + tn + fn).astype(np.float64)
        fp_rate = fp / (fp + tn)
        fn_rate = fn / (fn + tp)

        print("TEST SCORES")
        print('Acc %.4f ' % acc)
        print('Precision %.4f' % precisions)
        print('Recall %.4f' % recalls)
        print('F-1 %.4f' % f1s)
        print('fp rate ', fp_rate, 'fn rate', fn_rate)
