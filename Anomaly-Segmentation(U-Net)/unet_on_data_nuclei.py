
# coding: utf-8

# In[ ]:


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
# Remember to enable GPU
# %matplotlib inline

from models import *
from losses_and_metrics import *

import tensorflow as tf
from keras.optimizers import Adam

__READ_FROM_PICKLES__ = True

# %load_ext autoreload
# %autoreload 2


# In[18]:


def read_data_from_pickles(X_path="my_saved_files/X_train.npy", 
                           Y_path="my_saved_files/Y_train.npy", 
                           train_ids_path="my_saved_files/train_ids_ser.p"):
    
    
    X_train = np.load(X_path)
#     if X_train.max()>1.: X_train = X_train/255.
    Y_train = np.load(Y_path)
    train_ids_ser = pd.read_pickle(train_ids_path)
    
    X_test = np.load("my_saved_files/X_test.npy")
#     if X_test.max()>1.: X_test = X_test/255.
    test_ids_ser = pd.read_pickle("my_saved_files/test_ids_ser.p")

    print(X_train.shape, Y_train.shape, train_ids_ser.shape)
    
    return X_train, Y_train, train_ids_ser, X_test, test_ids_ser


# In[19]:


X, Y, train_ids_ser, X_, test_ids_ser = read_data_from_pickles()


# In[20]:


X.max(), Y.max()


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[22]:


unet = myUnetHP()
model = unet.build(n_depth_layers=6, n_init_filters=16)


# In[24]:


# Compile model:
model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=[mean_iou])


# In[25]:


# earlystopper = EarlyStopping(patience=100, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.33, patience=10, min_lr=1e-6)

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=200, 
                    callbacks=[checkpointer, reduce_lr])


# In[ ]:


model.save_weights("model-dsbowl2018-2.h5")

