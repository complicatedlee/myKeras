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
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report 
from sklearn.metrics import cohen_kappa_score 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.utils import plot_model
from keras.utils import plot_model, np_utils

from load_data_batches import load_data
import test
    

def mydense_net(img_rows, img_cols, color_type=1, num_classes=None):

    LT = L.Input(shape=(img_rows, img_cols, color_type), name = 'left_top')

    x, x_fc1=conv_block(LT, 5)

    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x_fc2=fc_block(x)

    x=L.Conv2D(16,(3, 3),activation='relu')(x) 
    x_fc3=fc_block(x)

    x, x_fc4=conv_block(x, 3)

    x=L.Conv2D(16,(3, 3),activation='relu')(x)
    x_fc5=fc_block(x) 

    # x=L.Conv2D(16,(3,3),padding='same')(x) 
    # x=L.Activation('relu')(x)
    # x_fc5=fc_block(x)

    x = L.concatenate([x_fc1, x_fc2, x_fc3, x_fc4, x_fc5])
    logits=L.Dense(num_classes,activation='softmax')(x)
    model=K.models.Model([LT],logits)
    opti = K.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-3)
    model.compile(optimizer=opti,loss='categorical_crossentropy',metrics=['acc'])
   
    return model

def conv_block(name, size):
    x=L.Conv2D(16,(size, size))(name) 
    x=L.Activation('relu')(x)
    x_fc=fc_block(x)
    x=L.BatchNormalization()(x)
    x=L.MaxPooling2D((2,2))(x)

    return x, x_fc

def fc_block(name):
    x = L.GlobalAveragePooling2D()(name)
    x = L.BatchNormalization()(x)
    x = L.Dense(512, activation='relu')(x)

    return x

def conv_dense(name):
    x_init1=L.Conv2D(16,(5,5),padding='same')(name) 
    x1=L.BatchNormalization()(x_init1)
    x1=L.Activation('relu')(x1)
    x=L.concatenate([x1, name])

    x_init2=L.Conv2D(16,(5,5),padding='same')(x) 
    x2=L.BatchNormalization()(x_init2)
    x2=L.Activation('relu')(x2)
    x=L.concatenate([x2,name,x1])

    x_init3=L.Conv2D(16,(5,5),padding='same')(x) 
    x3=L.BatchNormalization()(x_init3)
    x3=L.Activation('relu')(x3)

    return x




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
                        default='imagenet_models/1.h5',
                        help='Directory of model save')
    parser.add_argument('--mode',
                        type=int ,
                        default=0,
                        help='train or test mode')
    parser.add_argument('--epoch',
                        type=int ,
                        default=500,
                        help='train epochs')

    args=parser.parse_args()

    img_rows, img_cols = 32, 32# Resolution of inputs
    channel = 3
    num_classes = 4 # classes
    batch_size = 64

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    # x_train, y_train , x_test, y_test = load_data(img_cols,img_rows)
    
    (train_data, test_data, train_label, test_label) = test.assemble()
    x_train = train_data
    y_train = train_label
    x_test = np.vstack((test_data[0], test_data[1], test_data[2], test_data[3]))

    y_test = np.hstack((test_label[0], test_label[1], test_label[2], test_label[3]))

    if args.mode == 0:
        model_ckt = ModelCheckpoint(filepath=args.model_name, verbose=1, save_best_only=True)
        tensorbd=TensorBoard(log_dir='./log_lx',histogram_freq=0, write_graph=True, write_images=True)
        
        class_weights={}
        for c in xrange(num_classes):
            n=1.0*np.sum(y_train==c)
            item={c:n}
            class_weights.update(item)
        print class_weights    

        y_train = np_utils.to_categorical(y_train[:], num_classes)
        y_test = np_utils.to_categorical(y_test[:], num_classes)

        print('train data shape:{}'.format(x_train.shape))
        print('{} train sample'.format(x_train.shape[0]))
        #Load our model
        #model = multi_net(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
        model = mydense_net(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
        plot_model(model,to_file='model_uber.png',show_shapes=True)
        # model.load_weights('imagenet_models/myFcNet(save10).h5')
        # Start Fine-tuning
        # model.fit([x_train], y_train,
        #       batch_size=batch_size,
        #       epochs=args.epoch,
        #       shuffle=True,
        #       verbose=1,
        #       #validation_split=0.1
        #       validation_data=([x_test], y_test),
        #       callbacks=[model_ckt,tensorbd]
        #       )


        generator = ImageDataGenerator(
                                        rotation_range=30,
                                        # zca_whitening=True,
                                        horizontal_flip=True, 
                                        vertical_flip=True,
                                        # width_shift_range=0.1,
                                        # height_shift_range=0.1
                                    )

        generator.fit(x_train, seed=0)
        # generator = test.imageDataGenerator(X = train_data, Y = train_label , batch_size = batch_size)


        # # # Start Fine-tuning
        model.fit_generator(generator.flow(x_train, y_train, batch_size = batch_size),
                # class_weight=class_weights,
                steps_per_epoch=len(x_train)//batch_size, 
                epochs=args.epoch,
                validation_data=(x_test, y_test),
                callbacks=[model_ckt,tensorbd]
                )

        model.save(args.model_name)
    
    else:
        print('{} test sample'.format(y_test.shape[0]))
        #model = multi_net(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)      
        model = mydense_net(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
        model.load_weights(args.model_name)
        # Make predictions
        predictions_valid = model.predict([x_test], batch_size=batch_size, verbose=1) #, RT_test, LD_test, RD_test, C_test], batch_size=batch_size, verbose=1)
        
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

    




