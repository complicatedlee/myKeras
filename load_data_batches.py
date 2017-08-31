import cv2
import numpy as np
import os
import math

import scipy.io as sio


PATH = '/home/lixiang/myKeras/multi_input_cell/data/'
# TRAIN_DATA = 'multi_patch_train.mat'  #3 class
# TEST_DATA = 'multi_patch_test.mat'
TRAIN_DATA = 'cross_data1.mat'
TEST_DATA = 'cross_data2.mat'  #4 class

# def load_data(img_rows, img_cols):
#     """Loads dataset.

#     # Returns
#         Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
#     """

#     train_data = sio.loadmat(os.path.join(PATH + TRAIN_DATA))
#     test_data = sio.loadmat(os.path.join(PATH + TEST_DATA))
#     LT_train = np.array(train_data['left_top_train'])
#     RT_train = np.array(train_data['right_top_train'])
#     LD_train = np.array(train_data['left_down_train'])
#     RD_train = np.array(train_data['right_down_train'])
#     C_train = np.array(train_data['all_center_train'])

#     label_train = np.array(train_data['all_label_train'])

#     LT_test = np.array(test_data['left_top_test'])
#     RT_test = np.array(test_data['right_top_test'])
#     LD_test = np.array(test_data['left_down_test'])
#     RD_test = np.array(test_data['right_down_test'])
#     C_test = np.array(test_data['all_center_test'])

#     label_test = np.array(test_data['all_label_test'])

#     np.random.seed(0)
#     index=np.random.permutation(label_train.shape[0])

#     LT_train = LT_train.transpose(3,0,1,2)
#     RT_train = RT_train.transpose(3,0,1,2)
#     LD_train = LD_train.transpose(3,0,1,2)
#     RD_train = RD_train.transpose(3,0,1,2)
#     C_train = C_train.transpose(3,0,1,2)

#     LT_train = LT_train[index]
#     RT_train = RT_train[index]
#     LD_train = LD_train[index]
#     RD_train = RD_train[index]
#     C_train = C_train[index]
#     label_train = label_train[index]

#     LT_test = LT_test.transpose(3,0,1,2)
#     RT_test = RT_test.transpose(3,0,1,2)
#     LD_test = LD_test.transpose(3,0,1,2)
#     RD_test = RD_test.transpose(3,0,1,2)
#     C_test = C_test.transpose(3,0,1,2)

#     print "Resizing..."
#     LT_train = np.array([cv2.resize(img, (img_cols,img_rows)) for img in LT_train[:,:,:,:]])
#     LT_test = np.array([cv2.resize(img, (img_cols,img_rows)) for img in LT_test[:,:,:,:]])
#     print "Done..."

#     LT_train = sample_wise_standardization(LT_train)
#     LT_test = sample_wise_standardization(LT_test)

#     LT_train /= 255.
#     LT_test /= 255.
#     ##x_train,x_test format : (14282, 32, 32, 3)

#     return (LT_train, RT_train, LD_train, RD_train, C_train, label_train), (LT_test, RT_test, LD_test, RD_test, C_test, label_test)

def load_data(img_rows, img_cols):
    """Loads dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    train_data = sio.loadmat(os.path.join(PATH + TRAIN_DATA))
    test_data = sio.loadmat(os.path.join(PATH + TEST_DATA))
    x_train = np.array(train_data['train'])
    y_train = np.array(train_data['label'])
    x_test = np.array(test_data['test'])
    y_test = np.array(test_data['label'])
    

    np.random.seed(0)
    index=np.random.permutation(x_train.shape[0])

    x_train = x_train[index]
    y_train = y_train[:,index]

    y_train = y_train.transpose(1,0)
    y_test = y_test.transpose(1,0)

    print "Resizing..."
    x_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_train[:,:,:,:]])
    x_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_test[:,:,:,:]])

        
    # medium = np.array([cv2.resize(img, (64,64)) for img in x_train[:,:,:,:]])
    # large = np.array([cv2.resize(img, (96,96)) for img in x_train[:,:,:,:]])
    # medium_test = np.array([cv2.resize(img, (64,64)) for img in x_test[:,:,:,:]])
    # large_test = np.array([cv2.resize(img, (96,96)) for img in x_test[:,:,:,:]])
    print "Done..."

    x_train = sample_wise_standardization(x_train)
    x_test = sample_wise_standardization(x_test)

    # medium = sample_wise_standardization(medium)
    # medium_test = sample_wise_standardization(medium_test)
    # large = sample_wise_standardization(large)
    # large_test = sample_wise_standardization(large_test)
    #x_train,x_test format : (14282, 32, 32, 3)

    return x_train, y_train , x_test, y_test#, medium, medium_test, large, large_test

def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


if __name__ == "__main__":
    # (LT_train, RT_train, LD_train, RD_train, C_train, label_train), (LT_test, RT_test, LD_test, RD_test, C_test, label_test) = load_data()
    x_train, y_train , x_test, y_test = load_data()