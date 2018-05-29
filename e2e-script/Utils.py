
# coding: utf-8

# In[22]:


from imports import *
from glob import glob


# In[28]:


def from_binary_masks_to_colored_mask(masks):
    ### masks is a list of binary masks
    cmap=plt.cm.get_cmap('nipy_spectral')
    final = np.zeros((masks[0].shape[0], masks[0].shape[1], 3))
    for i in range(len(masks)):
        final[:,:,0] += masks[i]*cmap(i/len(masks))[0]
        final[:,:,1] += masks[i]*cmap(i/len(masks))[1]
        final[:,:,2] += masks[i]*cmap(i/len(masks))[2]
    return final


# In[29]:


def read_masks_from_dir(path):
    ### path comes with / at the end
    masks = []
    paths = glob(path + '*')
    for im_path in paths:
        masks.append(plt.imread(im_path))
    return masks


# In[35]:


if __name__=="__main__":
    path = "input/stage1_train/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/masks/"
    masks = read_masks_from_dir(path)
    colored_mask = from_binary_masks_to_colored_mask(masks)
    plt.imshow(colored_mask)


# In[39]:


def from_masks_to_binary_mask(masks):
    final = np.zeros((masks[0].shape[0], masks[0].shape[1]))
    for i in range(len(masks)):
        msk = masks[i]
        final += msk
    return final


# In[41]:


if __name__=="__main__":
    mask = from_masks_to_binary_mask(masks)
    plt.imshow(mask)


# In[43]:


def from_binary_mask_to_masks(mask):
    pass


# In[44]:


def from_mask_to_contour(mask):
    pass


# In[45]:


def from_masks_to_contours(masks):
    pass


# In[ ]:


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


# In[66]:


samples = glob("input/stage1_train/*")
samples[0]


# In[67]:


glob(samples[0] + "/images/*")


# In[77]:


images = []
binary_masks = []
colored_masks = []
for sample in samples[:3]:
    print("---------------------------------")
    img_f = glob(sample + "/images/*")
    images.append(plt.imread(img_f[0])[:, :, :3])
    masks = read_masks_from_dir(sample + "/masks/")
    print("img shape:", images[-1].shape)
    for mask in masks:
        print("\n", mask.shape)
    binary_masks.append(from_masks_to_binary_mask(masks))
    colored_masks.append(from_binary_masks_to_colored_mask(masks))
# binary_masks = np.stack(binary_masks)
# colored_masks = np.stack(colored_masks)


# In[71]:


plt.imshow(colored_masks[0])


# In[72]:


plt.imshow(binary_masks[0])


# In[ ]:


img_f

