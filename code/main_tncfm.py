from __future__ import absolute_import

__author__ = 'ChakkritTermritthikun: https://kaggle.com/kongpasom'

import numpy as np
np.random.seed(2017)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:12:01 2016

@author: sergi
"""

import os
import h5py
import numpy as np 
import keras.models as models
import cv2
import time
import threading

import theano
import theano.tensor as T

from keras.layers.advanced_activations import PReLU, LeakyReLU

from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution3D, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Layer, Permute
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from theano.compile.nanguardmode import NanGuardMode
from keras.preprocessing.image import ImageDataGenerator
THEANO_FLAGS=mode=NanGuardMode
np.random.seed(1337)
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential

from keras.optimizers import Adam
from theano.tensor.signal.conv import conv2d
from keras.layers.advanced_activations import ELU

#RESNET    
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot

    
# PRO UNET!
# https://github.com/EdwardTyantov/ultrasound-nerve-segmentation/blob/master/u_model.py
from keras.layers import Lambda   
    
def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(_input)

    return merge([shortcut, residual], mode="sum")


def inception_block(inputs, depth, batch_mode=0, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    
    c1_1 = Convolution2D(depth/4, 1, 1, init='he_normal', border_mode='same')(inputs)
    
    c2_1 = Convolution2D(depth/8*3, 1, 1, init='he_normal', border_mode='same')(inputs)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Convolution2D(depth/2, 1, 3, init='he_normal', border_mode='same')(c2_1)
        c2_2 = BatchNormalization(mode=batch_mode, axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Convolution2D(depth/2, 3, 1, init='he_normal', border_mode='same')(c2_2)
    else:
        c2_3 = Convolution2D(depth/2, 3, 3, init='he_normal', border_mode='same')(c2_1)
    
    c3_1 = Convolution2D(depth/16, 1, 1, init='he_normal', border_mode='same')(inputs)
    #missed batch norm
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Convolution2D(depth/8, 1, 5, init='he_normal', border_mode='same')(c3_1)
        c3_2 = BatchNormalization(mode=batch_mode, axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Convolution2D(depth/8, 5, 1, init='he_normal', border_mode='same')(c3_2)
    else:
        c3_3 = Convolution2D(depth/8, 5, 5, init='he_normal', border_mode='same')(c3_1)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), border_mode='same')(inputs)
    c4_2 = Convolution2D(depth/8, 1, 1, init='he_normal', border_mode='same')(p4_1)
    
    res = merge([c1_1, c2_3, c3_3, c4_2], mode='concat', concat_axis=1)
    res = BatchNormalization(mode=batch_mode, axis=1)(res)
    res = actv()(res)
    return res
    

def rblock(inputs, num, depth, scale=0.1):    
    residual = Convolution2D(depth, num, num, border_mode='same')(inputs)
    residual = BatchNormalization(mode=2, axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res) 
    

def NConvolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    def f(_input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                              border_mode=border_mode)(_input)
        norm = BatchNormalization(mode=2, axis=1)(conv)
        return ELU()(norm)

    return f

def BNA(_input):
    inputs_norm = BatchNormalization(mode=2, axis=1)(_input)
    return ELU()(inputs_norm)

def reduction_a(inputs, k=64, l=64, m=96, n=96):
    "35x35 -> 17x17"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
    
    conv2 = Convolution2D(n, 3, 3, subsample=(2,2), border_mode='same')(inputs_norm)
    
    conv3_1 = NConvolution2D(k, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv3_2 = NConvolution2D(l, 3, 3, subsample=(1,1), border_mode='same')(conv3_1)
    conv3_2 = Convolution2D(m, 3, 3, subsample=(2,2), border_mode='same')(conv3_2)
    
    res = merge([pool1, conv2, conv3_2], mode='concat', concat_axis=1)
    return res


def reduction_b(inputs):
    "17x17 -> 8x8"
    inputs_norm = BNA(inputs)
    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
    #
    conv2_1 = NConvolution2D(64, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv2_2 = Convolution2D(96, 3, 3, subsample=(2,2), border_mode='same')(conv2_1)
    #
    conv3_1 = NConvolution2D(64, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv3_2 = Convolution2D(72, 3, 3, subsample=(2,2), border_mode='same')(conv3_1)
    #
    conv4_1 = NConvolution2D(64, 1, 1, subsample=(1,1), border_mode='same')(inputs_norm)
    conv4_2 = NConvolution2D(72, 3, 3, subsample=(1,1), border_mode='same')(conv4_1)
    conv4_3 = Convolution2D(80, 3, 3, subsample=(2,2), border_mode='same')(conv4_2)
    #
    res = merge([pool1, conv2_2, conv3_2, conv4_3], mode='concat', concat_axis=1)
    return res

def myhypercolumn(shp=(3,128,128), weights_path=''):
    splitted = True
    act = 'relu' #'elu'
    
    inputs = Input(shp, name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)
    
    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
    pool4 = NConvolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)
    
    res = inception_block(conv5, 8, batch_mode=2, splitted=splitted, activation=act)
    
    '''
    #Hypercolumns
    hc_conv5 = UpSampling2D(size=(16, 16))(conv5) #8x8, f = 16
    hc_conv4 = UpSampling2D(size=(8, 8))(conv4) #16x16, f = 8
    hc_conv3 = UpSampling2D(size=(4, 4))(conv3) #32x32, f = 4
    hc_conv2 = UpSampling2D(size=(2, 2))(conv2) #64x64, f = 2
    
    hc = merge([conv1, hc_conv2, hc_conv3, hc_conv4, hc_conv5], mode='concat', concat_axis=1) #(None, 992, 128, 128)

    #From (None, 992, 128, 128) to 3x128x128
    #hc_red_conv1 = inception_block(hc, 128, batch_mode=2, splitted=splitted, activation=act)
    #hc_red_conv2 = inception_block(hc_red_conv1, 64, batch_mode=2, splitted=splitted, activation=act)
    hc_red_zpad1 = ZeroPadding2D((1,1))(hc)
    hc_red_conv1 = Convolution2D(128, 3, 3, init='he_normal', activation=act)(hc_red_zpad1)
    hc_red_zpad2 = ZeroPadding2D((1,1))(hc_red_conv1)
    hc_red_conv2 = Convolution2D(64, 3, 3, init='he_normal', activation=act)(hc_red_zpad2)
    hc_red_conv3 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid', name='aux_output')(hc_red_conv2)
    
    #Deconvolution process
    after_conv4 = rblock(conv4, 1, 256)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)
    
    after_conv3 = rblock(conv3, 1, 128)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), after_conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, splitted=splitted, activation=act)
    conv7 = Dropout(0.5)(conv7)
    
    after_conv2 = rblock(conv2, 1, 64)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), after_conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up8, 64, batch_mode=2, splitted=splitted, activation=act)
    conv8 = Dropout(0.5)(conv8)
    
    after_conv1 = rblock(conv1, 1, 32)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), after_conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, splitted=splitted, activation=act)
    conv9 = Dropout(0.5)(conv9)
    conv10 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid', name='main_output')(conv9)

    model = Model(input=inputs, output=[conv10, hc_red_conv3]) #output=[conv10, aux_out]
    
    if weights_path <> '':
        print('-- Loading weights...')
        model.load_weights(weights_path)
    
    model.compile(optimizer='Adam', loss={'main_output': 'categorical_crossentropy', 'aux_output': 'mse'}, loss_weights={'main_output': 1., 'aux_output': 0.2})
    '''
    
    model = Model(input=inputs, output=[conv5])
    model.compile(loss='categorical_crossentropy', optimizer='adam')    

    return model

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (128, 128))#, cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model():
    '''
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 48, 48), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    
    model.add(Flatten())
    model.add(Dense(96, activation='relu',init='he_uniform'))
    model.add(Dropout(0.515))
    model.add(Dense(16, activation='relu',init='he_uniform'))
    model.add(Dropout(0.515))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.898, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    '''
    model = myhypercolumn()

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 24
    nb_epoch = 15
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
    