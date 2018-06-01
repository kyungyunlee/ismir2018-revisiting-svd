import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.signal import medfilt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
from load_data import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()


# RandomForestClassifier Lehner 2013
def train(savefile, gridsearch=False):
    X_train, y_train = load_xy_data(None, FEAT_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'train')

    X_train_shuffled, _, y_train_shuffled, _ = train_test_split(X_train, y_train, test_size=0.0, random_state=42)

    # perform grid search 
    if gridsearch:
        X_val, y_val = load_xy_data(None, FEAT_JAMENDO_DIR, JAMENDO_LABEL_DIR, 'valid')
        X_train = X_train.reshape((X_train.shape[0], -1))
        print(X_train.shape, y_train.shape)
        X_val = X_val.reshape((X_val.shape[0], -1))

        X_cv = np.concatenate((X_train, X_val), axis=0)
        y_cv = np.concatenate((y_train, y_val), axis=0)
        N = X_train.shape[0]
        M = X_val.shape[0]
        print(N, M)

        idxs = [-1 for i in range(N)] + [0 for i in range(M)]
        print(len(idxs), N + M)

        cv_iter = PredefinedSplit(test_fold=idxs)
        # cv_iter = StratifiedKFold(n_splits=10, shuffle=True,random_state=1)
        param_grid = {
            'max_features': [20],
            'min_samples_leaf': [10],
            'min_samples_split': [10, 30, 50],
            'n_estimators': [50, 100, 200, 500]
        }
        clf = RandomForestClassifier(bootstrap=True, random_state=1, criterion='gini', n_jobs=4)
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv_iter, verbose=2)
        grid_search.fit(X_cv, y_cv)

        print(grid_search.best_params_)
        pickle.dump(grid_search.best_params_, open('best_params.pkl', 'wb'))
        pickle.dump(grid_search.cv_results_, open('results.pkl', 'wb'))
        pickle.dump(grid_search.best_estimator_, open(savefile, 'wb'))

    else:
        clf = RandomForestClassifier(bootstrap=True, random_state=1, min_samples_leaf=10, min_samples_split=50,
                                     n_estimators=128, n_jobs=4)

        clf.fit(X_train, y_train)
        pickle.dump(clf, open(savefile, 'wb'))


if __name__ == '__main__':
    # best model 20180531-1.sav
    savefile = './weights/' + args.model_name + '.sav'
    train(savefile, gridsearch=False)
    print("training finished")
