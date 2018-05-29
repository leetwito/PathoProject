
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

import tensorflow as tf
from keras.optimizers import Adam

__READ_FROM_PICKLES__ = True


# In[2]:


def mean_iou(y_true, y_pred, t = 0.5):
    y_pred_ = tf.to_int32(y_pred > t)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, num_classes=2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


# In[ ]:


def mean_iou_offline(y_true_in, y_pred_in, t = 0.5):
    yt = tf.placeholder(dtype=tf.bool, shape=y_true_in.shape)
    yp= tf.placeholder(dtype=tf.float32, shape=y_pred_in.shape)
    y_true = tf.to_int32(yt)
    y_pred = tf.to_int32(yp > t)
    score, up_opt = tf.metrics.mean_iou(yt, y_pred, num_classes=2)
#     print(score.get_shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
        feed_dict = {yt:y_true_in, yp:y_pred_in}
        c_mat_out, score_out = sess.run([up_opt, score], feed_dict=feed_dict)
    
#     K.get_session().run(tf.global_variables_initializer())
#     score = K.get_session().run(score)
#     with tf.control_dependencies([]):
#         score = tf.identity(score)
    return c_mat_out[1,1] / (c_mat_out[1,1] + c_mat_out[0,1] + c_mat_out[1,0])


# In[3]:


def mean_iou_np(y_true, y_pred, t = 0.5):
    y_pred = y_pred>t
    score = (y_pred*y_true).sum() / (y_pred+y_true).sum()
    return score

