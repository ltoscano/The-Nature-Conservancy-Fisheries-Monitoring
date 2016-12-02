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

import unet as unet

RESOLUTION = 128


def unison_shuffled_copies(a, b):
    # Randomize two vectors
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_im_cv2(path, img_type='normal'):
    # Read the image
    if img_type == 'mask':
        img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(path)
        
    # resize
    resized = cv2.resize(img, (RESOLUTION, RESOLUTION)) #, cv2.INTER_LINEAR)
    
    # normalize
    resized = resized/255.0   
    return resized

    
def get_train():
    print('-- Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    
    train_files = []
    train_labels = []
    mask_files = []
    
    for fld in folders:
        index = folders.index(fld)
        print('load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            #Image
            flbase = os.path.basename(fl) #train_files.append(fl)    
            img = get_im_cv2(fl,'normal')
            
            #Mask
            if fld == 'NoF':
                mask = np.zeros([RESOLUTION,RESOLUTION])
            else:
                mask_path = fl.replace('train', 'train_masks') #mask_files.append(mask_path)
                mask = get_im_cv2(mask_path,'mask')         
                
            train_files.append(img)
            train_labels.append(flbase)
            mask_files.append(mask)
    
    #Convert to numpy array
    train_files = np.array(train_files).transpose(0,3,1,2) #3312,3,128,128
    mask_files = np.expand_dims(np.array(mask_files), axis=1) #3312,1,128,128
    
    # Randomize the vector that contains the images. Y must be ranzomized too
    # train_files, mask_files = unison_shuffled_copies(train_files,mask_files)
    return train_files, train_labels, mask_files


def check_img(train_files, mask_files, idx=0):
     #Test how it works
    cv2.imwrite('img.jpg', train_files[idx].transpose(1,2,0)*255)
    cv2.imwrite('mask.jpg', mask_files[idx].transpose(1,2,0)*255)   

    
def print_predictions(general_predictions, num_outputs=10):
    for idx in range(num_outputs):       
        output_img = general_predictions[idx]

        oname = os.path.join('../output/predictions', str(idx)+'_prediction.jpg')
        cv2.imwrite(oname, output_img.transpose(1,2,0)*255)
        
        oname = os.path.join('../output/predictions', str(idx)+'_gt.jpg')
        cv2.imwrite(oname, mask_files[idx].transpose(1,2,0)*255)


if __name__ == '__main__':    
    from keras.callbacks import ModelCheckpoint

    # Prepare the training data
    train_files, train_labels, mask_files = get_train()

    # Check if the images are correct
    # check_img(train_files, mask_files, 40)
    
    # Train a unet
    model = unet.train_mask_detector(RESOLUTION, RESOLUTION, '../output/weights/unet_mask_checkpoint.hdf5')
    
    '''
    # Fit the data
    if not os.path.isfile('../output/weights') and not os.path.isdir('../output/weights'):
        os.mkdir('../output/weights')
    kfold_weights_path = os.path.join('../output/weights', 'mask_checkpoint' + '.hdf5' )
    callbacks = [
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]
    model.fit(train_files, mask_files, batch_size=32, nb_epoch=100, verbose=1, validation_split=0.2, callbacks=callbacks)
    '''
    
    # Predict
    num_outputs = 10
    general_predictions = model.predict(np.array(train_files[0:num_outputs]))
    print_predictions(general_predictions, num_outputs)


        
