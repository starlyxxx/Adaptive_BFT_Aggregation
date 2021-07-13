import sys
import os
import numpy as np
import pickle
import keras
from keras import backend as K
K.set_floatx('float64')


"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

class Car10():

    def load_batch(fpath, label_key='labels'):
        """Internal utility for parsing CIFAR data.

        # Arguments
            fpath: path the file to parse.
            label_key: key for label data in the retrieve
                dictionary.

        # Returns
            A tuple `(data, labels)`.
        """
        with open(fpath, 'rb') as f:
            if sys.version_info < (3,):
                d = pickle.load(f)
            else:
                d = pickle.load(f, encoding='bytes')
                # decode utf8
                d_decoded = {}
                for k, v in d.items():
                    d_decoded[k.decode('utf8')] = v
                d = d_decoded
        data = d['data']
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels


    def load_data():
        """Loads CIFAR10 dataset.

        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        path = './cifar-10-batches-py'

        num_train_samples = 50000

        x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train_local = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
            y_train_local[(i - 1) * 10000: i * 10000]) = Car10.load_batch(fpath)

        fpath = os.path.join(path, 'test_batch')
        x_test_local, y_test_local = Car10.load_batch(fpath)

        y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
        y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

        if K.image_data_format() == 'channels_last':
            x_train_local = x_train_local.transpose(0, 2, 3, 1)
            x_test_local = x_test_local.transpose(0, 2, 3, 1)

        return (x_train_local, y_train_local), (x_test_local, y_test_local)

    def post_process(x, y):
        if K.image_data_format() == 'channels_first':
            sample_shape = (1, ) + x.shape[1:]
        else:
            sample_shape = x.shape[1:] + (1, )
        # x = x.reshape((x.shape[0],) + sample_shape)

        y_vec = keras.utils.to_categorical(y, 10)
        return x / 255.,y_vec

    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6,.3,.1)):
        return ((self.x_train, self.y_train),
                (self.x_test, self.y_test),
                (self.x_valid, self.y_valid))

    #def load_data():
    def __init__(self):
    #if __name__ == "__main__":
        """show it works"""

        (train_data, train_labels), (test_data, test_labels) = Car10.load_data()
        
        total_train_size, total_test_size = train_data.shape[0], test_data.shape[0]

        total_valid_size = int(total_train_size * .3)
        total_train_size = int(total_train_size * .7)

        train_sample_idx = np.random.choice(total_train_size, 800,replace=True)
        valid_sample_idx = np.random.choice(range(total_train_size, total_train_size + total_valid_size), 400, replace=True)
        test_sample_idx = np.random.choice(total_test_size, 100, replace=True)
        
        
        # train_sample_idx = np.random.choice(train_data.shape[0], 900,replace=True)
        # test_sample_idx = np.random.choice(test_data.shape[0], 400,replace=True)
        self.x_train, self.y_train = Car10.post_process(
                    train_data[train_sample_idx], train_labels[train_sample_idx])
        self.x_valid, self.y_valid= Car10.post_process(
                    train_data[valid_sample_idx], train_labels[valid_sample_idx])
        self.x_test, self.y_test = Car10.post_process(
                    test_data[test_sample_idx], test_labels[test_sample_idx])             
        
        # return (train_data[0:800][0:][0:][0:],train_labels[0:800][0:]),(train_data[800:900][0:][0:][0:],train_labels[800:900][0:]),(test_data[0:400][0:][0:][0:],test_labels[0:400][0:])
        
        # print("Train data: ", train_data.shape)
        # print("Train filenames: ", train_filenames.shape)
        # print("Train labels: ", train_labels.shape)
        # print("Test data: ", test_data.shape)
        # print("Test filenames: ", test_filenames.shape)
        # print("Test labels: ", test_labels.shape)
        # print("Label names: ", label_names.shape)

        # print("!!!!!!",train_data[0:800][0:][0:][0:].shape)
        # # Don't forget that the label_names and filesnames are in binary and need conversion if used.

        # # display some random training images in a 25x25 grid
        # num_plot = 5
        # f, ax = plt.subplots(num_plot, num_plot)
        # for m in range(num_plot):
        #     for n in range(num_plot):
        #         idx = np.random.randint(0, train_data.shape[0])
        #         ax[m, n].imshow(train_data[idx])
        #         ax[m, n].get_xaxis().set_visible(False)
        #         ax[m, n].get_yaxis().set_visible(False)
        # f.subplots_adjust(hspace=0.1)
        # f.subplots_adjust(wspace=0)
        # plt.show()