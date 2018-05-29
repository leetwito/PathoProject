
# coding: utf-8

# In[21]:


import os
import csv
import pandas as pd
import cv2
import numpy as np
from skimage.morphology import label
import datetime


# In[18]:


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# In[19]:


topDir = "C:/Users/leetw/PycharmProjects/PathoProject/Anomaly-Segmentation(U-Net)/input"

test_path = os.path.join(topDir, 'stage1_test')   #path to test data file/folder
Y_hat = cv2.imread("C:/Users/leetw/Desktop/yin-and-yang.png") # todo: should be list of predictions matching test samples

# Apply Run-Length Encoding on our Y_hat_upscaled
new_test_ids = []
rles = []
for n, id_ in enumerate(os.listdir(test_path)):
    rle = list(prob_to_rles(Y_hat))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
len(new_test_ids)  #note that for each test_image, we can have multiple entries of encoded pixels


# In[25]:


# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
print('Submission output to: {}/working/sub-{}.csv'.format(topDir, timestamp))
sub.to_csv(topDir+"/working/sub-{}.csv".format(timestamp), index=False)


# In[24]:


# Have a look at our submission pandas dataframe
sub.head()

