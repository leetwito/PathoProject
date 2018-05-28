
# coding: utf-8

# In[1]:


from imports import *
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import OneHotEncoder


# In[2]:


x = np.load('x_for_clustering.npy')
# x.shape
xx = []
for im in x:
#     print(im.shape)
    xx.append(cv2.resize(im, (224, 224)))
xx = np.stack(xx)
# xx.shape


# In[3]:


yy = pd.read_csv('data_for_clustering.csv', header=None)[1]
yy.value_counts()
w = 1/yy.value_counts()
w


# In[4]:


le = OneHotEncoder()
y = le.fit_transform(np.expand_dims(yy, axis=1)).todense()


# In[5]:


x = x[yy.index]


# In[6]:


# y.shape
# y.todense()


# In[7]:


plt.imshow(xx[0])


# In[8]:


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))


# In[9]:


top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(11, activation='sigmoid'))
model = Model(input=vgg16.input, output=top_model(vgg16.output))


# In[10]:


model.summary()


# In[11]:


data_gen_args_img = dict(samplewise_center=True,
                     samplewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2, 
                     fill_mode="reflect")
image_datagen = ImageDataGenerator(**data_gen_args_img)
seed = 7
BATCH_SIZE = 8
LR_MIN = 1e-9
LR_INIT = 1e-3
image_datagen.fit(x, augment=True, seed=seed)
image_generator = image_datagen.flow(x, y, seed=seed, batch_size=BATCH_SIZE, shuffle=True)


# In[12]:


# Compile model:
model.compile(optimizer=Adam(lr = LR_INIT), loss='binary_crossentropy', metrics=["accuracy"])


# In[13]:


checkpointer = ModelCheckpoint('model-nuclei2018-1.h5', verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=LR_MIN, verbose=1) # search "learning rate"
def sched(epoch, lr):
    if epoch%20==0:
        lr = lr*0.8
    return lr
reduce_lr = keras.callbacks.LearningRateScheduler(sched, verbose=1)


# In[ ]:


results = model.fit_generator(image_generator, epochs=10000, callbacks=[checkpointer, reduce_lr],
                              steps_per_epoch=x.shape[0]//BATCH_SIZE, class_weight=w,
                              verbose=1)


# In[ ]:


model.saveweights('vgg_classifier_27.5')


# In[ ]:


res = np.argmax(model.predict(x), axis=1)


# In[ ]:


res

