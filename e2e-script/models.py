
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join
import glob
import sys
import random
import warnings
from tqdm import tqdm
import itertools
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras.layers import merge
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import initializers, layers, models
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.initializers import glorot_normal
# Remember to enable GPU
# %matplotlib inline

import tensorflow as tf
from keras.optimizers import Adam

__READ_FROM_PICKLES__ = True


# In[ ]:


class myUnetHP(object):
    def build(self, n_depth_layers, n_init_filters, IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=3, verbose=1, initializer=glorot_normal, x_max=1., dropouts_frac=None):
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="l0_input")
        s = Lambda(lambda x: x / x_max, name="l0_normalize") (inputs)
        tmp = s

        # encoder layers:
        encoder_layers = dict()
        n_filters = n_init_filters
        for i in range(n_depth_layers):
#             print(n_filters)
            encoder_layers[i+1] = dict()
            tmp = encoder_layers[i+1]["1c"] = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name="enc_l%d_1c"%(i+1), kernel_initializer=initializer()) (tmp)
            tmp = encoder_layers[i+1]["2c"] = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name="enc_l%d_2c"%(i+1), kernel_initializer=initializer()) (tmp)
            tmp = encoder_layers[i+1]["3p"] = MaxPooling2D((2, 2), name="enc_l%d_3p"%(i+1)) (tmp)
            n_filters = 2*n_filters
            encoder = tmp

        # central layers:
        central_convs = dict()
#         print(n_filters)
        tmp = central_convs[1] = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name="mid_1conv", kernel_initializer=initializer()) (tmp)
        tmp = central_convs[2] = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name="mid_2conv", kernel_initializer=initializer()) (tmp)

        # # decoder layers:
        decoder_layers = dict()
        for i in range(n_depth_layers):
            n_filters = n_filters//2
#             print(n_filters)
            decoder_layers[i+1] = dict()
            tmp = decoder_layers[i+1]["1u"] = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name="dec_l%d_1u"%(i+1), kernel_initializer=initializer()) (tmp)
            tmp = decoder_layers[i+1]["2concat"] = concatenate([tmp, encoder_layers[n_depth_layers-(i)]["2c"]], name="dec_l%d_2concat"%(i+1))
            tmp = decoder_layers[i+1]["3c"] = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name="dec_l%d_3c"%(i+1), kernel_initializer=initializer()) (tmp)
            tmp = decoder_layers[i+1]["4c"] = Conv2D(n_filters, (3, 3), activation='relu', padding='same', name="dec_l%d_4c"%(i+1), kernel_initializer=initializer()) (tmp)


        outputs = Conv2D(1, (1, 1), activation='sigmoid') (tmp)

        model = Model(inputs=[inputs], outputs=[outputs])
        if verbose>0: 
            model.summary()
        
        self.model = model
        return  self.model

