
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
# Remember to enable GPU
# %matplotlib inline

from models import *
from losses_and_metrics import *
from generators import *

import tensorflow as tf
from keras.optimizers import Adam, RMSprop

from glob import glob
from tqdm import tqdm

__READ_FROM_PICKLES__ = True

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass


# In[2]:


def plot_image(img):
    parms_dict = {}
    if len(img.shape)==4 and img.shape[0]==1:
        img = img.squeeze(axis=0)
    if len(img.shape)==3 and img.shape[-1]==1:
        img = img.squeeze(axis=-1)
        parms_dict["cmap"] = plt.get_cmap("gray")

    if img.max()>1:
        img = img/255.
    plt.imshow(img, **parms_dict)


# In[3]:


def plot_img_mask_maskPred(img, msk, mskP=None, figsize=(16,8)):
    fig = plt.figure(figsize=figsize)
    
    plt.subplot(131)
    plot_image(img)
    plt.subplot(132)
    plot_image(msk)
    if mskP is not None:
        plt.subplot(133)
        plot_image(mskP)


# In[4]:


def read_data_from_pickles(X_path="my_saved_files/X_train.npy", 
                           Y_path="my_saved_files/Y_train.npy", 
                           train_ids_path="my_saved_files/train_ids_ser.p"):
    
    
    X_train = np.load(X_path)
    Y_train = np.load(Y_path)
    train_ids_ser = pd.read_pickle(train_ids_path)
    
    X_test = np.load("my_saved_files/X_test.npy")
    test_ids_ser = pd.read_pickle("my_saved_files/test_ids_ser.p")

    print(X_train.shape, Y_train.shape, train_ids_ser.shape)
    
    return X_train, Y_train, train_ids_ser, X_test, test_ids_ser


# In[21]:


# we create two instances with the same arguments
# data_gen_args = dict(rotation_range=90.,
#                      width_shift_range=0.5,
#                      height_shift_range=0.5,
#                      zoom_range=0.2, 
#                      fill_mode="constant",
#                      cval=1)

data_gen_args_img = dict(samplewise_center=True,
                     samplewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2, 
                     fill_mode="reflect")

data_gen_args_msk = dict(rotation_range=90.,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2, 
                     fill_mode="reflect")


def create_generators_pair(X, Y, seed=0):
    image_datagen = ImageDataGenerator(**data_gen_args_img)
    mask_datagen = ImageDataGenerator(**data_gen_args_msk)
    
    image_datagen.fit(X, augment=True, rounds=2, seed=seed)
    mask_datagen.fit(Y, augment=True, rounds=2, seed=seed)

    # image_generator = image_datagen.flow_from_directory( # in the future use this function to load data
    #     'data/images',
    #     class_mode=None,
    #     seed=seed)

    # mask_generator = mask_datagen.flow_from_directory(
    #     'data/masks',
    #     class_mode=None,
    #     seed=seed)

    image_generator = image_datagen.flow(X, y=Y seed=seed, batch_size=BATCH_SIZE, shuffle=True)
#     mask_generator = mask_datagen.flow(Y, seed=seed, batch_size=BATCH_SIZE, shuffle=True) ###

    # combine generators into one which yields image and masks
#     train_generator = zip(image_generator, mask_generator)###
#     return image_generator, mask_generator, train_generator###

    return image_generator


# In[22]:


# IMG_HEIGHT=960
# IMG_WIDTH=1280
# IMG_CHANNELS=3

UNET_DEPTH = 6
UNET_INIT_FILTERS = 16

BATCH_SIZE = 8
LR_INIT = 1e-3
LR_MIN = 1e-9


# In[23]:


# BATCH_SIZE = 1


# ### Read and Split Data Patho:

# In[35]:


X, Y, train_ids_ser, X_, test_ids_ser = read_data_from_pickles()
print(X.shape, X.max(), X.min())
print(Y.shape, np.unique(Y))
X.shape


# #### Split to train/test/val

# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)


# In[26]:


# i = 6
# X_train = np.expand_dims(X[i], axis=0)
# Y_train = np.expand_dims(Y[i], axis=0)

# X_val= np.expand_dims(X[i], axis=0)
# Y_val = np.expand_dims(Y[i], axis=0)

# X_test = np.expand_dims(X[i], axis=0)
# Y_test = np.expand_dims(Y[i], axis=0)


# ### Create Generator w/o Augmentations:

# In[27]:


# image_generator_train, mask_generator_train, generator_train = create_generators_pair(X_train, Y_train) ###
# image_generator_val, mask_generator_val, generator_val = create_generators_pair(X_val, Y_val)###


# In[27]:


image_generator_train = create_generators_pair(X_train, Y_train)
image_generator_val, mask_generator_val, generator_val = create_generators_pair(X_val, Y_val)


# In[28]:


# image_generator_train.next()


# In[39]:


def compre_generators_image_mask(image_generator, mask_generator, n=10):
    for i in range(n):
        x = image_generator.next()[0]
        x = (x-x.min())/(x-x.min()).max()
        y = mask_generator.next()[0]
        print(x.shape, x.max(), x.min())
        print(y.shape, y.max(), y.min(), len(np.unique(y))) # todo: problem - y isn't binary
        fig = plt.figure()
        plot_img_mask_maskPred(x, y)


# In[40]:
 

compre_generators_image_mask(image_generator_train, mask_generator_train)


# In[41]:


compre_generators_image_mask(image_generator_val, mask_generator_val)


# In[ ]:


# data_gen_args = dict(
# #                         fill_mode="constant", # the new pixels filling method (options: ...) 
# #                         cval=255,             # for fill_mode="constant", the value those pixels receives.
# #                         height_shift_range=0.5, # randomly shifts the image up/down up to 30% of the image height 
#                     )


# gen = ImageDataGenerator(**data_gen_args)
# gen.fit(X_train) # not really required here, until some augmentations rely on all data-set (X) data
# gen = gen.flow(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=False)

# gen_val = ImageDataGenerator(**data_gen_args)
# gen_val.fit(X_train) # not really required here, until some augmentations rely on all data-set (X) data
# gen_val = gen_val.flow(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)

# gen_test = ImageDataGenerator(**data_gen_args)
# gen_test.fit(X_train) # not really required here, until some augmentations rely on all data-set (X) data
# gen_test = gen_test.flow(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


# # how to use this iterator:
# x, y = train_generator.next()
# x_val, y_val = gen_val.next()
# print(x.shape, y.shape, x.max(), x.min(), y.max(), y.mean(), (x_val!=x).sum(), (y_val!=y).sum())


# In[ ]:


# x, y = .next()
# plot_img_mask_maskPred(x[0], y[0])
# x, y = gen_val.next()
# plot_img_mask_maskPred(x[0], y[0])
# x, y = gen_test.next()
# plot_img_mask_maskPred(x[0], y[0])


# ### Model: Create, Compile, Fit

# In[8]:


unet = myUnetHP()
model = unet.build(n_depth_layers=UNET_DEPTH, n_init_filters=UNET_INIT_FILTERS, x_max=1.)


# In[39]:


# Compile model:
model.compile(optimizer=Adam(lr = LR_INIT), loss='binary_crossentropy', metrics=[mean_iou])


# In[40]:


checkpointer = ModelCheckpoint('model-nuclei2018-1.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30, min_lr=LR_MIN, verbose=1) # search "learning rate"


# In[41]:


# model_weights_path = "model-nuclei2018-2-dph%d_flts%d.h5"%(UNET_DEPTH, UNET_INIT_FILTERS)
model_weights_path = "model-nuclei2018-1.h5"
# model_weights_path = "model-nuclei2018-3-dph%d_flts%d_epochs30000.h5"%(UNET_DEPTH, UNET_INIT_FILTERS)
# if os.path.isfile(model_weights_path):
#     model.load_weights("model_weights_path")


# In[ ]:


results = model.fit_generator(generator_train, epochs=10000, validation_data=generator_val, callbacks=[checkpointer, reduce_lr],
                              steps_per_epoch=X_train.shape[0]//BATCH_SIZE, 
                              validation_steps=X_val.shape[0]//BATCH_SIZE, 
                              verbose=1)


# In[ ]:


model.save_weights("model-nuclei2018-3-dph%d_flts%d_epochs10000_20180524.h5"%(UNET_DEPTH, UNET_INIT_FILTERS))


# In[ ]:


# Y_pred = model.predict_generator(image_generator_train, steps=X_train.shape[0]//BATCH_SIZE)
i = 0
Y_pred = model.predict(np.expand_dims(X_train[i], axis=0))


# In[ ]:


Y_pred.max()


# In[ ]:


mean_iou_offline(y_true_in=np.expand_dims(Y_train[i], axis=0), y_pred_in=Y_pred)


# In[ ]:


plot_img_mask_maskPred(X_train[i], Y_train[i], Y_pred[0])


# ### Evaluate Model Visually

# #### Train

# In[9]:


model_weights_path = "model-nuclei2018-3-dph%d_flts%d_epochs10000.h5"%(UNET_DEPTH, UNET_INIT_FILTERS)
model.load_weights(model_weights_path)


# In[10]:


Y_pred = model.predict(X_train)


# In[ ]:


mean_ious = []  # todo: run in batches ?
for i in tqdm(range(len(X_train))):
    mean_ious.append(mean_iou_offline(y_true_in=np.expand_dims(Y_train[i], axis=0), y_pred_in=np.expand_dims(Y_pred[i], axis=0)))


# In[ ]:


ser = pd.Series(mean_ious)
bins = np.arange(0, 1.05, 0.05)
ser.hist(bins=bins, figsize=(8, 8))


# In[ ]:


ts = 0.7

for i in ser[ser<ts].index:
    plot_img_mask_maskPred(X_train[i], Y_train[i], Y_pred[i])

