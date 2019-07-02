import os
import sys
import numpy as np
import pickle
import keras
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from keras.models import load_model
import tensorflow as tf
from scipy.signal import medfilt
import argparse
from load_data import *
from model import *
from config_rnn import *

np.random.seed(0)

# set gpu number 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--input', type=str)
args = parser.parse_args()


def main():
    # load model
    model_name = args.model_name
    loaded_model = load_model('./weights/rnn_' + model_name + '.h5')
    print("loaded model")
    print(loaded_model.summary())

  
    input_mel = process_single_audio(args.input)
    
    total_x = []
    total_y = []
    
    x = input_mel
    for i in range(0, x.shape[1] - RNN_INPUT_SIZE, 1):
        x_segment = x[:, i: i + RNN_INPUT_SIZE]
        y_segment = y[i: i + RNN_INPUT_SIZE]
        total_x.append(x_segment)
        total_y.append(y_segment)

    total_x = np.array(total_x)
    total_y = np.array(total_y)
    try:
        mean_std = np.load("train_mean_std_" + model_name + '.npy')
        mean = mean_std[0]
        std = mean_std[1]
    except:
        print("mean, std not found")
        sys.exit()

    total_x_norm = (total_x - mean) / std
    total_x_norm = np.swapaxes(total_x_norm, 1, 2)

    x_test = total_x_norm
    y_pred = loaded_model.predict(x_test, verbose=1) # Shape=(total_frames,)
    
    print(y_pred)
	return y_pred
	
	
if __name__ == "__main__":
	main()
	