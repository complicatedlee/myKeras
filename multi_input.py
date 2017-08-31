# -*- coding: utf-8 -*-

# In[1]:
import keras as K
import keras.layers as L
import tensorflow as tf
import scipy.io as sio
import argparse,os
import numpy as np
import h5py
import time
import sys
import cv2
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report 
from sklearn.metrics import cohen_kappa_score 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.utils import plot_model
from keras.utils import plot_model, np_utils

from load_data_batches import load_data
import test


def conv_block(name, kernel_size):
    x=L.Conv2D(16,(kernel_size, kernel_size))(name) 
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.MaxPooling2D((2,2))(x)

    return x

def small_model(img_rows, img_cols, color_type, num_classes=None):
    small = L.Input(shape=(img_rows, img_cols, color_type), name = 'small')
 
    ####################
    x=conv_block(small, 5)
    x=L.Conv2D(36,(5, 5),activation='relu')(small) 
    x=L.Conv2D(48,(3, 3),activation='relu')(x) 

    x=L.Flatten()(x)
    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    
    logits=L.Dense(num_classes,activation='softmax',name = 'softmax1')(x)
    model=K.models.Model([small],logits)
    opti = K.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-3)
    model.compile(optimizer=opti,loss='categorical_crossentropy',metrics=['acc'])
   
    return model


def medium_model(medium, num_classes=None):
    # medium = L.Input(shape=(img_rows, img_cols, color_type), name = 'medium')
    x=conv_block(medium, 5)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=conv_block(x, 3)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 

    x=L.Flatten()(x)
    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    
    logits=L.Dense(num_classes,activation='softmax',name = 'softmax2')(x)
    model=K.models.Model([medium],logits)
    opti = K.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-3)
    model.compile(optimizer=opti,loss='categorical_crossentropy',metrics=['acc'])
   
    return model

def large_model(large, num_classes=None):
    ######################
    # large = L.Input(shape=(img_rows, img_cols, color_type), name = 'large')

    x=conv_block(large, 5)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=conv_block(x, 3)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 

    x=L.Flatten()(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    
    logits=L.Dense(num_classes,activation='softmax',name = 'softmax3')(x)
    model=K.models.Model([large],logits)
    opti = K.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-3)
    model.compile(optimizer=opti,loss='categorical_crossentropy',metrics=['acc'])
   
    return model


def multi_net(img_rows, img_cols, color_type, num_classes=None):
    small = L.Input(shape=(img_rows, img_cols, color_type), name = 'small')
 
    x=conv_block(small, 5)
    x=L.Conv2D(36,(5, 5),activation='relu')(small) 
    x=L.Conv2D(48,(3, 3),activation='relu')(x) 
    x=L.Flatten()(x)
    x1=L.Dense(256)(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    
    logits1=L.Dense(num_classes,activation='softmax')(x)


    medium = L.Input(shape=(img_rows*2, img_cols*2, color_type), name = 'medium')
    x=conv_block(medium, 5)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=conv_block(x, 3)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Flatten()(x)
    x2=L.Dense(256)(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    
    logits2=L.Dense(num_classes,activation='softmax')(x)
    
    large = L.Input(shape=(img_rows*3, img_cols*3, color_type), name = 'large')

    x=conv_block(large, 5)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=conv_block(x, 3)
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x=L.Flatten()(x)
    x3=L.Dense(256)(x)
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    x = L.Dense(512, activation='relu')(x)
    x = L.Dropout(0.4)(x)    
    
    logits3=L.Dense(num_classes,activation='softmax')(x)

    # # combine all patch
    # merge0=L.concatenate([x1, x2, x3],axis=-1)
    # merge1=L.Dense(1024)(merge0)
    # merge2=L.Activation('relu')(merge1)
    # merge2 = L.Dropout(0.4)(merge2) 

    # merge3=L.Dense(512)(merge2)
    # merge3=L.Activation('relu')(merge3)
    # merge3 = L.Dropout(0.4)(merge3) 

    # logits=L.Dense(num_classes,activation='softmax')(merge0)
    logits=L.average([logits1, logits2, logits3]) 


    new_model = K.models.Model([small, medium, large], logits)
    sgd=K.optimizers.SGD(lr=0.001,momentum=0.99,decay=1e-4)
    new_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['acc'])
    return new_model


def trainable_net(img_rows, img_cols, color_type, num_classes=None):

    small_input = L.Input(shape=(img_rows, img_cols, channel))
    small_net = small_model(small_input, num_classes = num_classes)
    small_net.load_weights('imagenet_models/multi_cnn_small.h5')
    # small_net.layers.pop()
    small_net.trainable=True
    small_output=small_net.layers[-1].output

    medium_input = L.Input(shape=(img_rows*2, img_cols*2, channel))
    medium_net = medium_model(medium_input, num_classes = num_classes)
    medium_net.load_weights('imagenet_models/multi_cnn_medium.h5')
    # medium_net.layers.pop()
    medium_net.trainable=False
    medium_output=medium_net.layers[-1].output

    large_input = L.Input(shape=(img_rows*3, img_cols*3, channel))
    large_net = large_model(large_input, num_classes = num_classes)
    large_net.load_weights('imagenet_models/multi_cnn_large.h5')
    # large_net.layers.pop()
    large_net.trainable=False
    large_output=large_net.layers[-1].output

    logits=L.average([small_output, medium_output, large_output]) 

    new_model = K.models.Model([small_input, medium_input, large_input], logits)
    sgd=K.optimizers.SGD(lr=0.001,momentum=0.99,decay=1e-4)
    new_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['acc'])
    return new_model


def eval(predication,labels):
    """
    evaluate test score
    """
    num=labels.shape[0]
    count=0
    for i in xrange(num):
        if(np.argmax(predication[i])==labels[i]):
            count+=1
    return 100.0*count/num


if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='imagenet_models/multi_cnn_small.h5',
                        help='Directory of model save')
    parser.add_argument('--mode',
                        type=int ,
                        default=0,
                        help='train or test mode')
    parser.add_argument('--epoch',
                        type=int ,
                        default=300,
                        help='train epochs')

    args=parser.parse_args()

    img_rows, img_cols = 32, 32 # Resolution of inputs
    channel = 3
    num_classes = 4
    batch_size = 512

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    x_train, y_train , x_test, y_test = load_data(img_cols,img_rows)
    
    # (train_data, test_data, train_label, test_label) = test.assemble()


    if args.mode == 0:
        model_ckt = ModelCheckpoint(filepath=args.model_name, verbose=1, save_best_only=True)
        tensorbd=TensorBoard(log_dir='./log_lx',histogram_freq=0, write_graph=True, write_images=True)

        y_train = np_utils.to_categorical(y_train[:], num_classes)
        y_test = np_utils.to_categorical(y_test[:], num_classes)

        print('train data shape:{}'.format(x_train.shape))
        print('{} train sample'.format(x_train.shape[0]))
        #Load our model
        # model = multi_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
        model = small_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
        plot_model(model,to_file='multi_model.png',show_shapes=True)
        # model.load_weights('imagenet_models/cnn_6.h5')

        medium = np.array([cv2.resize(img, (64,64)) for img in x_train[:,:,:,:]])
        large = np.array([cv2.resize(img, (96,96)) for img in x_train[:,:,:,:]])
        medium_test = np.array([cv2.resize(img, (64,64)) for img in x_test[:,:,:,:]])
        large_test = np.array([cv2.resize(img, (96,96)) for img in x_test[:,:,:,:]])

        generator = ImageDataGenerator(
                                        rotation_range=30,
                                        # zca_whitening=True,
                                        horizontal_flip=True, 
                                        vertical_flip=True,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1
                                    )

        generator.fit(x_train, seed=0)


        # # # Start Fine-tuning
        model.fit_generator(generator.flow(x_train, y_train, batch_size = batch_size),
                steps_per_epoch=len(x_train)//batch_size, 
                epochs=args.epoch,
                validation_data=(x_test, y_test),
                callbacks=[model_ckt,tensorbd]
                )

        # # Start Fine-tuning
        # model.fit([x_train, medium, large], y_train,
        #       batch_size=batch_size,
        #       epochs=args.epoch,
        #       shuffle=True,
        #       verbose=1,
        #       #validation_split=0.1
        #       validation_data=([x_test], y_test), #, medium_test, large_test], y_test),
        #       callbacks=[model_ckt,tensorbd]
        #       )

        model.save(args.model_name)
    
    else:
        print('{} test sample'.format(y_test.shape[0]))
        model = multi_net(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)      
        
        model.load_weights(args.model_name)
        # Make predictions

        medium_test = np.array([cv2.resize(img, (64,64)) for img in x_test[:,:,:,:]])
        large_test = np.array([cv2.resize(img, (96,96)) for img in x_test[:,:,:,:]])
        predictions_valid = model.predict([x_test, medium_test, large_test], batch_size=batch_size, verbose=1) #, RT_test, LD_test, RD_test, C_test], batch_size=batch_size, verbose=1)
        
        print(predictions_valid.shape,y_test.shape)

        # print score
        print('OA: {}%'.format(eval(predictions_valid,y_test)))

        # generate confusion_matrix
        prediction=np.asarray(predictions_valid)
        pred=np.argmax(prediction,axis=1)
        pred=np.asarray(pred,dtype=np.int8)
        print confusion_matrix(y_test,pred)

        # generate accuracy
        print classification_report(y_test, pred)
        print cohen_kappa_score(y_test,pred)

    



