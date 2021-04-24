#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import resnet50, VGG16, InceptionV3, Xception
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D, AveragePooling2D, \
    Activation

# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Model
import tensorflow as tf
import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# In[4]:


basedir = "./DATASET"  # here below the train and validation data


# In[5]:


def removeCorruptedImages(path):
    for filename in os.listdir(path):
        try:
            img = Image.open(os.path.join(path, filename))
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(os.path.join(path, filename))


# In[6]:


traindir = os.path.join(basedir, 'TRAIN')  # root for training
validdir = os.path.join(basedir, 'TEST')  # root for testing

# In[7]:


removeCorruptedImages(os.path.join(traindir, 'downdog'))
removeCorruptedImages(os.path.join(traindir, 'goddess'))
removeCorruptedImages(os.path.join(traindir, 'plank'))
removeCorruptedImages(os.path.join(traindir, 'tree'))
removeCorruptedImages(os.path.join(traindir, 'warrior2'))
removeCorruptedImages(os.path.join(validdir, 'downdog'))
removeCorruptedImages(os.path.join(validdir, 'goddess'))
removeCorruptedImages(os.path.join(validdir, 'plank'))
removeCorruptedImages(os.path.join(validdir, 'tree'))
removeCorruptedImages(os.path.join(validdir, 'warrior2'))

# In[ ]:


# In[8]:


train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# In[10]:


bx = 16
train_datagen = train_datagen.flow_from_directory(
    directory='./DATASET/TRAIN',
    target_size=(300, 300),
    batch_size=bx,
    shuffle=True,
    class_mode='categorical')

val_datagen = test_datagen.flow_from_directory(
    directory='./DATASET/TEST',
    target_size=(300, 300),
    batch_size=bx,
    shuffle=False,
    class_mode='categorical')

# In[12]:


base_model = Xception(weights='./PRETRAINED_MODELS/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      include_top=False, input_shape=(300, 300, 3))

# In[13]:


model = Sequential()
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Activation('softmax'))

model = Model(inputs=base_model.input, outputs=model(base_model.output))

optimizers = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
losss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)
model.compile(loss=losss,
              optimizer=optimizers
              , metrics=['accuracy'])

model.summary()

# In[ ]:


Hist = model.fit_generator(generator=train_datagen,
                           validation_data=val_datagen,
                           epochs=15
                           )

# In[12]:


model.save("Yoga_Detection1.hdf5")

# In[13]:


plt.figure(0)
plt.plot(Hist.history['loss'], 'g')
plt.plot(Hist.history['val_loss'], 'b')
plt.plot(Hist.history['accuracy'], 'r')
plt.plot(Hist.history['val_accuracy'], 'black')
plt.show()

# In[ ]:
