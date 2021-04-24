import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.optimizers as Opti
import keras.models as Modl
import keras.losses as Loss
import keras.layers as Layr
import keras.preprocessing as PreP
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from IPython.display import display

import h5py

# input vars
batch_size = 32
img_dims = 64

# Data Augmentation
train_data = PreP.image.ImageDataGenerator(rescale=1. / 255)
test_data = PreP.image.ImageDataGenerator(rescale=1. / 255)

# train_2_data = PreP.image.ImageDataGenerator(rescale=1. / 255,
#                                              shear_range=0.2,
#                                              zoom_range=0.2,
#                                              rotation_range=20,
#                                              horizontal_flip=True)
# test_2_data = PreP.image.ImageDataGenerator(rescale=1. / 255)

# keep stuff clean
# find . -type f -name ".*" -exec rm -f {} \;
classes = 5
training_set = train_data.flow_from_directory('images/Project/DATASET/TRAIN', # 5 classes
                                              target_size=(img_dims, img_dims),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_set = test_data.flow_from_directory('images/Project/DATASET/TEST', # 5 classes
                                         target_size=(img_dims, img_dims),
                                         batch_size=batch_size,
                                         class_mode='categorical')

#classes = 10
# training_set = train_data.flow_from_directory('images/training_set', # 10 classes
#                                                  target_size=(img_dims, img_dims),
#                                                  batch_size=batch_size,
#                                                  class_mode='categorical')

# test_set = test_data.flow_from_directory('images/test_set', # 10 classes
#                                             target_size=(img_dims, img_dims),
#                                             batch_size=batch_size,
#                                             class_mode='categorical')

# CNN model 
model = Modl.Sequential()
model.add(Layr.Conv2D(32, (3, 3), input_shape=(img_dims, img_dims, 3), activation='relu'))
model.add(Layr.MaxPooling2D(pool_size=(2, 2)))
model.add(Layr.Conv2D(32, (3, 3), activation='relu'))
model.add(Layr.MaxPooling2D(pool_size=(2, 2)))
model.add(Layr.Flatten())
model.add(Layr.Dense(units=img_dims * 2, activation='relu'))

# ensure this matches the number of folders in the ImageDataGenerator
model.add(Layr.Dense(units=classes, activation='softmax'))  # number of classes

optimizer = Opti.SGD(learning_rate=0.002, momentum=0.9, nesterov=True)
loss = Loss.CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# Fit model to data
h = model.fit(training_set,
              steps_per_epoch=729 // batch_size,  # number of training set images, 729
              epochs=5,
              validation_data=test_set,
              validation_steps=109 // batch_size)  # number of test set images, 229

model.save('models.tmp/model_multiclass10.h5')  # save model

img_data = np.random.random(size=(32, 32, 3))
img = PreP.image.array_to_img(img_data)
plt.figure(0)
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

# key check
print(h.history.keys())
# Plot cool stuff with matlab
plt.rcParams["figure.figsize"] = [8, 4]
plt.subplot(1, 2, 1)
plt.plot(h.history['accuracy'], label='Training')
plt.plot(h.history['val_accuracy'], label='Validation')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h.history['loss'], label='Training')
plt.plot(h.history['val_loss'], label='Validation')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

# plt.figure(1)
# plt.plot(h.history['loss'], 'g')
# plt.plot(h.history['val_loss'], 'b')
# plt.plot(h.history['accuracy'], 'r')
# plt.plot(h.history['val_accuracy'], 'black')
# plt.show()
