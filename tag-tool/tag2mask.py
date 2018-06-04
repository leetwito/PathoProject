'''
convert tag png file to binary mask.
also removes the unnecessary layers.
run this code after running pdn2png.exe command line (see README.txt file for detailed information)
modify DIR_PATH variable
PathoProject, Lee & Shlomi 2018
'''

import cv2
import numpy as np
from scipy import misc
from glob import glob
import matplotlib.pyplot as plt
import os

def crateSubDirs(DIR_PATH):
    if not os.path.exists(DIR_PATH + '\\images'):
        os.makedirs(DIR_PATH + '\\images')
    if not os.path.exists(DIR_PATH + '\\tags'):
        os.makedirs(DIR_PATH + '\\tags')

def getMaskFromTag(path):
    im = plt.imread(path)
    im = im[:, :, 3]
    im_thresh = im > 0.5
    return im_thresh

DIR_PATH = 'C:\\Users\\Lee Twito\\Desktop\\ADENOMA 1 - 159'
imgs_paths = glob(DIR_PATH+'\\*.png')
crateSubDirs(DIR_PATH)

for path in imgs_paths:
    if path.endswith('Background.png'):
        # print(os.path.dirname(path)+'\\images\\'+os.path.basename(path))
        os.rename(path, DIR_PATH+'\\images\\'+os.path.basename(path))
    else:
        mask = getMaskFromTag(path)
        plt.imsave(DIR_PATH+'\\tags\\'+os.path.basename(path), mask, cmap=plt.get_cmap("gray"))
        os.remove(path)
        # plt.imshow(mask)
        # plt.show()
        # print(os.path.dirname(path)+'\\tags\\'+os.path.basename(path))
        # os.rename(path, os.path.dirname(path)+'\\tags\\'+os.path.basename(path))