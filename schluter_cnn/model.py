from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU


def Schluter_CNN(dropout_rate):
    ''' Model from Schluter et al. (2015 ISMIR Data Augmentation paper)
    Data input size : (input_frame_size, n_melbins, 1) == (115, 80, 1)
    Args:
        dropout_rate : dropout rate at the dense layer

    Return:
        None
    '''
    input_shape = (80, 115, 1)
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), name='conv1', padding='valid', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(LeakyReLU(0.01))
    model.add(Conv2D(32, (3, 3), name='conv2', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D((3, 3), strides=(3, 3)))

    model.add(Conv2D(128, (3, 3), name='conv3', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(Conv2D(64, (3, 3), name='conv4', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D((3, 3), strides=(3, 3)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    model = Schluter_CNN(0.5)
    print(model.summary())
