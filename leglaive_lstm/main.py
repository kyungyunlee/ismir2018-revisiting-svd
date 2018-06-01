import os
import numpy as np
import keras
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score
import h5py
import argparse

from load_data import *
from model import *
from config_rnn import *

np.random.seed(0)

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()


class Score_History(keras.callbacks.Callback):
    ''' Keras callback function to calculat f1, precision, recall scores after each epoch '''

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]
        y_pred[y_pred >= THRESHOLD] = True
        y_pred[y_pred < THRESHOLD] = False
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        f1_ = f1_score(y_true, y_pred, average='binary')
        precision_ = precision_score(y_true, y_pred, average='binary')
        recall_ = recall_score(y_true, y_pred, average='binary')
        print("f1 : {}, precision : {}, recall : {}".format(f1_, precision_, recall_))
        return


def train(model_save_path):
    model = Leglaive_RNN(timesteps=RNN_INPUT_SIZE)

    opt = Adam(lr=LEARNING_RATE)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_weights_only=False,
                                 save_best_only=True,
                                 mode='auto')
    earlyStopping = EarlyStopping(monitor='val_acc',
                                  patience=7,
                                  verbose=1,
                                  mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.8,
                                  patience=5,
                                  verbose=1,
                                  min_lr=1e-8)

    x_train, y_train = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, args.model_name, 'train')
    x_val, y_val = load_xy_data(None, MEL_JAMENDO_DIR, JAMENDO_LABEL_DIR, args.model_name, 'valid')

    histories = Score_History()

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
              callbacks=[checkpoint, earlyStopping, reduce_lr, histories], shuffle=True, validation_data=(x_val, y_val))

    # model.save("model_" + args.model_name + ".h5")
    print("Finished training")


if __name__ == '__main__':
    ckpt_name = './weights/rnn_' + args.model_name + '.h5'
    train(ckpt_name)
