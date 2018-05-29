
# coding: utf-8

# In[1]:


from imports import *


# In[2]:


cmap=plt.cm.get_cmap('prism')
cmap=plt.cm.get_cmap('Paired')
cmap=plt.cm.get_cmap('nipy_spectral')


# In[3]:


x = np.array([0,1])
for i in range(10):
    y = i*x
    plt.plot(x,y, color=cmap(0.1*i))


# #### Plot Masks

# In[4]:


from glob import glob


# In[7]:


masks = glob("input/stage1_train/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/masks/*")


# In[24]:


final = np.zeros((256, 256, 3))
for i in range(len(masks)):
    msk = plt.imread(masks[i])
    final[:,:,0] += msk*cmap(i/len(masks))[0]
    final[:,:,1] += msk*cmap(i/len(masks))[1]
    final[:,:,2] += msk*cmap(i/len(masks))[2]


# #### build overall mask

# In[41]:


final = np.zeros((256, 256))
for i in range(len(masks)):
    msk = plt.imread(masks[i])
    final += msk


# In[42]:


plt.imshow(final, cmap=plt.get_cmap("gray"))


# #### From segmented to masks

# In[44]:


import scipy 


# In[45]:


labeled_array, num_features = scipy.ndimage.label(final)
num_features


# In[51]:


masks = []
for i in range(1, num_features+1):
    masks.append(labeled_array==i)


# In[53]:


# for mask in masks:
#     plt.figure()
#     plt.imshow(mask)

