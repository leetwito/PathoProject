
# coding: utf-8

# In[1]:


'''
create contours and semantic masks for data
'''


# In[4]:


import matplotlib.pyplot as plt
from glob import glob
import numpy as np


# In[5]:


imgs_paths = glob('C:/Users/leetw/PycharmProjects/PathoProject/Anomaly-Segmentation(U-Net)/input/stage1_train/*/images/*.png')
masks_paths = glob('C:/Users/leetw/PycharmProjects/PathoProject/Anomaly-Segmentation(U-Net)/input/stage1_train/*/masks/*.png')
print(len(imgs_paths)) # should be 664

imgs = []
masks = {}
contours = []
for im_idx, im_path in enumerate(imgs_paths):
    image = plt.imread(im_path)[:,:,:3]
    imgs.append(image)
    masks[im_idx] = np.zeros((image.shape[0], image.shape[1]))
#     print(masks[im_idx].shape)
#     print(image.shape)
    plt.imshow(image)
    plt.show
    for mask_idx, mask_path in enumerate(masks_paths):
#         print(plt.imread(mask).shape)
#         print(masks[im_idx].shape) 
        try:
            masks[im_idx] += mask_idx * plt.imread(mask_path)
        except:
            print(mask_path)
            print(plt.imread(mask_path).shape)
    
    print(masks_idx.dtype)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(masks[im_idx])
    plt.show
    
        
# In[17]:


mask = cv2.imread(masks_paths[11], 0)
print(mask.shape)
cv2.imshow('masks', mask)
cv2.waitKey[0]


# In[20]:


import cv2
ret,thresh = cv2.threshold(mask,128,255,0)
plt.imshow(thresh)
im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
plt.drawContours(im, countours, -1, (0,255,0), 3)
cnt = contours[0]
M = cv2.moments(cnt)
print(M)


# In[46]:


from skimage import measure, morphology 
from scipy import ndimage


# In[54]:


con = measure.find_contours(mask, 0.8)[0].astype('int')
disk_kernel = morphology.disk(1)

con_image = np.zeros(mask.shape)

con_image[con[:,0], con[:,1]] = 1
con_image = ndimage.binary_dilation(con_image, disk_kernel)
# con = cv2.dilate(con, disk_kernel, iterations = 1)
plt.imshow(con_image)


# In[37]:


con.shape

