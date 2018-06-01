import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow import set_random_seed
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
import h5py
import argparse

from load_data import *
from model import *
from config_cnn import *

set_random_seed(0)
np.random.seed(0)
# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
print("model name", args)

# for optimizing threshold value 
thresholds = np.arange(1, 100) / 100.0
thresholds = thresholds[:, np.newaxis]

best_sofar = 0.0


class Score_History(keras.callbacks.Callback):
    ''' Callback function for calculating f1, precision, recall scores after every epoch
    '''

    def on_epoch_end(self, epoch, logs={}):
        # scores for predicting vocal segments
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]

        '''
        y_pred[y_pred >= THRESHOLD] = True
        y_pred[y_pred < THRESHOLD] = False
        '''
        print(y_pred.shape, y_true.shape)
        pred = y_pred[:, 0] >= thresholds
        pred = pred.astype(int)
        true = y_true[:, 0] >= thresholds
        true = true.astype(int)
        print(pred.shape, true.shape)

        accs = (((y_true.shape[0] - np.sum(np.abs(pred - true), axis=1))) / float(y_true.shape[0])) * 100.0
        print(accs.shape)
        best = np.argmax(accs)
        print("best threshold, acc", thresholds[best], accs[best])

        accuracy = accs[best]
        f1_ = f1_score(true[best], pred[best], average='binary')
        precision_ = precision_score(true[best], pred[best], average='binary')
        recall_ = recall_score(true[best], pred[best], average='binary')
        '''
        f1_ = f1_score(y_true, y_pred, average='binary')
        precision_ = precision_score(y_true, y_pred, average='binary')
        recall_ = recall_score(y_true, y_pred, average='binary')
        '''

        print("acc: {}, f1 : {}, precision : {}, recall : {}".format(accuracy, f1_, precision_, recall_))
        return


def train():
    model = Schluter_CNN(DROPOUT_RATE)
    model_save_path = './weights/cnn_' + args.model_name + '.h5'

    opt = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
    # opt = Adam(lr=LEARNING_RATE)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_weights_only=False,
                                 save_best_only=True,
                                 mode='auto')
    earlyStopping = EarlyStopping(monitor='val_acc',
                                  patience=EARLY_STOPPING,
                                  verbose=1,
                                  mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.2,
                                  patience=REDUCE_LR,
                                  verbose=1,
                                  min_lr=1e-8)

    x_train, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, args.model_name, 'train')
    x_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, args.model_name, 'valid')
    print("train, valid data loaded", x_val.shape, y_val.shape)

    # my call back function 
    histories = Score_History()

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              callbacks=[checkpoint, earlyStopping, reduce_lr, histories], shuffle=True, validation_data=(x_val, y_val))

    print("Finished!")


if __name__ == '__main__':
    train()
