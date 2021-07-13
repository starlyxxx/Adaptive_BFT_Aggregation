from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras
import numpy as np
import random
from keras import backend as K


def generate_fake_MNIST(train_data,test_data):

    K.set_floatx('float32')
    # Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
    
    # X_train, y_train = train_data
    # X_test, y_test = test_data

    mnistdata = np.load('./mnist.npz')
    x_train = mnistdata['x_train']
    x_test = mnistdata['x_test']

    X_train = x_train.reshape(60000, 28, 28, 1)
    X_test = x_test.reshape(10000, 28, 28, 1)
    X_train = X_train.astype('float64')/255
    X_test = X_test.astype('float64')/255

    # fake_data = load_car10.Car10().fake_non_iid_data('','','')
    # train_data, test_data, valid_data = fake_data
    # X_test, global_y_test = test_data
    # X_train, global_y_train = train_data

    z_dim = 100

    # Generator
    adam = Adam(lr=0.0002, beta_1=0.5)
    samples = []

    g = Sequential()
    g.add(Dense(7*7*112, input_dim=z_dim))
    g.add(Reshape((7, 7, 112)))
    g.add(BatchNormalization())
    g.add(Activation(LeakyReLU(alpha=0.2)))
    g.add(Conv2DTranspose(56, 5, strides=2, padding='same'))
    g.add(BatchNormalization())
    g.add(Activation(LeakyReLU(alpha=0.2)))
    g.add(Conv2DTranspose(1, 5, strides=2, padding='same', activation='sigmoid'))
    g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    K.set_floatx('float64')
    BATCH_SIZE = 400

    print('Batch size:', BATCH_SIZE)
    
    for i in range(100):  # tqdm_notebook(range(batchCount), leave=False):
        # Create a batch by drawing random index numbers from the training set
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
        image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1)
        # Create noise vectors for the generator
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
        
        # Generate the images from the noise
        generated_images = g.predict(noise)
        samples.append(generated_images)
        X = np.concatenate((image_batch, generated_images))
        # Create labels
        y = np.zeros(2*BATCH_SIZE)
        y[:BATCH_SIZE] = 0.9  # One-sided label smoothing
        
    y = keras.utils.to_categorical(y, 10)

    X = X.astype('float64')
    y = y.astype('float64')

    return X, y

if __name__ == "__main__":
    x,y=generate_fake_MNIST(1,2)
    y.dtype = "float64"